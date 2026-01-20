"""
Evaluation Script for FinBERT NER Model
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from .train_ner import NERDataset
from .config import PROCESSED_DATA_DIR, MODEL_DIR, MODEL_CONFIG, LABEL2ID, ID2LABEL


class NEREvaluator:
    """Evaluate trained NER model on test set"""
    
    def __init__(self, model_path: Path = MODEL_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def evaluate(self, test_file: Path = PROCESSED_DATA_DIR / "test.conll"):
        """Evaluate on test set"""
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        
        print(f"\nLoading test data from: {test_file}")
        
        # Load test dataset
        test_dataset = NERDataset(
            test_file,
            self.tokenizer,
            MODEL_CONFIG["max_length"]
        )
        
        # DataLoader with collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=MODEL_CONFIG["max_length"]
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=MODEL_CONFIG["batch_size"],
            collate_fn=data_collator
        )
        
        print(f"\nEvaluating on {len(test_dataset)} samples...")
        
        # Collect predictions
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Convert to label strings (full sequences)
                for pred, label, mask in zip(predictions, labels, attention_mask):
                    pred_labels = []
                    true_labels = []
                    
                    for p, l, m in zip(pred, label, mask):
                        if m == 1 and l != -100:
                            pred_labels.append(ID2LABEL.get(p.item(), "O"))
                            true_labels.append(ID2LABEL.get(l.item(), "O"))
                    
                    if pred_labels and true_labels:
                        all_predictions.append(pred_labels)
                        all_labels.append(true_labels)
        
        # Calculate metrics
        print("\n" + "="*80)
        print("TEST SET RESULTS")
        print("="*80)
        
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        
        print(f"\nOverall Metrics:")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(all_labels, all_predictions))
        
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }


if __name__ == "__main__":
    evaluator = NEREvaluator()
    evaluator.evaluate()
