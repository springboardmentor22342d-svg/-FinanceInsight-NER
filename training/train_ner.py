"""
FinBERT NER Training Script
Train Named Entity Recognition model on financial articles
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from .config import (
    PROCESSED_DATA_DIR, MODEL_DIR, MODEL_CONFIG,
    LABEL2ID, ID2LABEL, EARLY_STOPPING_PATIENCE
)


class NERDataset(Dataset):
    """Custom Dataset for NER training from CoNLL format"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int):
        """Load CoNLL file directly"""
        print(f"Loading dataset from: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        self.data = self._load_conll(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"  ✓ Loaded {len(self.data)} sentences")
    
    def _load_conll(self, filepath: Path) -> List[Dict]:
        """Load CoNLL format file (2-column: token label)"""
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Empty line = end of sentence
                if not line:
                    if current_tokens:
                        sentences.append({
                            "tokens": current_tokens,
                            "labels": current_labels
                        })
                        current_tokens = []
                        current_labels = []
                    continue
                
                # Parse: token label
                parts = line.split()
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[1])
        
        # Add last sentence if exists
        if current_tokens:
            sentences.append({
                "tokens": current_tokens,
                "labels": current_labels
            })
        
        return sentences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item["labels"]
        
        # Convert labels to IDs
        label_ids = [LABEL2ID.get(label, LABEL2ID.get("O", 0)) for label in labels]
        
        # Tokenize with word alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Align labels with subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(label_ids[word_idx])
            else:
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }



class FinBERTNERTrainer:
    """FinBERT NER Training Pipeline"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"\nLoading FinBERT with {len(LABEL2ID)} NER labels...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["base_model"])
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_CONFIG["base_model"],
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        print(f"✓ Model loaded successfully with {len(LABEL2ID)} labels")
        
        # Load datasets
        print("\nLoading datasets...")
        self.train_dataset = NERDataset(
            PROCESSED_DATA_DIR / "train.conll",
            self.tokenizer,
            MODEL_CONFIG["max_length"]
        )
        
        self.val_dataset = NERDataset(
            PROCESSED_DATA_DIR / "val.conll",
            self.tokenizer,
            MODEL_CONFIG["max_length"]
        )
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=MODEL_CONFIG["batch_size"],
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=MODEL_CONFIG["batch_size"]
        )
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=MODEL_CONFIG["learning_rate"],
            weight_decay=MODEL_CONFIG["weight_decay"]
        )
        
        total_steps = len(self.train_loader) * MODEL_CONFIG["num_epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=MODEL_CONFIG["warmup_steps"],
            num_training_steps=total_steps
        )
        
        self.best_f1 = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MODEL_CONFIG["max_grad_norm"])
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Convert to label strings (skip padding and special tokens)
                for pred, label in zip(predictions, labels):
                    pred_labels = []
                    true_labels = []
                    
                    for p, l in zip(pred, label):
                        if l != -100:
                            pred_label = ID2LABEL.get(p.item(), "O")
                            true_label = ID2LABEL.get(l.item(), "O")
                            pred_labels.append(pred_label)
                            true_labels.append(true_label)
                    
                    if pred_labels:
                        all_predictions.append(pred_labels)
                        all_labels.append(true_labels)
        
        # Calculate metrics
        if all_predictions and all_labels:
            f1 = f1_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)
        else:
            f1 = precision = recall = 0.0
        
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("STARTING FINBERT NER TRAINING")
        print("="*80)
        print(f"\nTrain samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Epochs: {MODEL_CONFIG['num_epochs']}")
        print(f"Batch size: {MODEL_CONFIG['batch_size']}")
        print(f"Learning rate: {MODEL_CONFIG['learning_rate']}")
        print(f"Device: {self.device}\n")
        
        for epoch in range(MODEL_CONFIG["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{MODEL_CONFIG['num_epochs']}")
            print("-" * 80)
            
            # Train
            train_loss = self.train_epoch()
            print(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            metrics = self.evaluate()
            print(f"Validation metrics:")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            
            # Save best model
            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                
                print(f"✓ New best F1: {self.best_f1:.4f} - Saving model...")
                self.save_model()
            else:
                print(f"  No improvement.")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best F1 Score: {self.best_f1:.4f}")
        print(f"Model saved to: {MODEL_DIR}")
    
    def save_model(self):
        """Save model and tokenizer"""
        self.model.save_pretrained(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)



if __name__ == "__main__":
    trainer = FinBERTNERTrainer()
    trainer.train()




