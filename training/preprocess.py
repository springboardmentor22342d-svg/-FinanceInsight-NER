"""
Data Preprocessing Pipeline
Load Label Studio JSON annotations and create train/val/test splits
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from transformers import AutoTokenizer
from tqdm import tqdm

from .config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_CONFIG,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


class FinanceDataPreprocessor:
    """
    Preprocess Label Studio JSON annotations for NER training
    Handles character-level entity annotations
    """
    
    def __init__(self, tokenizer_name: str = MODEL_CONFIG["base_model"]):
        """Initialize preprocessor with tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = MODEL_CONFIG["max_length"]
        self.detected_labels = set()
    
    def load_label_studio_json(self, filepath: Path) -> List[Dict]:
        """
        Load Label Studio JSON export
        
        Expected structure:
        [
          {
            "id": 372,
            "data": {"text": "..."},
            "annotations": [{
              "result": [
                {"value": {"start": 5, "end": 9, "text": "gold", "labels": ["INSTRUMENT"]}}
              ]
            }]
          },
          ...
        ]
        """
        print(f"\nLoading Label Studio annotations from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it's a list
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list, got {type(data)}")
        
        print(f"✓ Loaded {len(data)} annotated articles")
        return data
    
    def convert_to_tokens(self, articles: List[Dict]) -> List[Dict]:
        """
        Convert character-level annotations to token-level IOB2 format
        """
        all_sentences = []
        
        print("\nConverting annotations to token-level format...")
        
        for article in tqdm(articles, desc="Processing articles"):
            # Get article text
            text = article.get('data', {}).get('text', '')
            
            if not text:
                continue
            
            # Get annotations
            annotations = article.get('annotations', [])
            
            if not annotations or not annotations[0].get('result'):
                # No annotations - skip or process as plain text
                continue
            
            # Get first annotation result
            entities = annotations[0].get('result', [])
            
            # Create entity spans dictionary: (start, end) -> label
            entity_spans = {}
            for entity in entities:
                value = entity.get('value', {})
                start = value.get('start')
                end = value.get('end')
                labels = value.get('labels', [])
                
                if start is not None and end is not None and labels:
                    entity_spans[(start, end)] = labels[0]
                    self.detected_labels.add(f"B-{labels[0]}")
                    self.detected_labels.add(f"I-{labels[0]}")
            
            self.detected_labels.add("O")
            
            # Split text into sentences and tokenize
            sentences = self._split_into_sentences(text, entity_spans)
            all_sentences.extend(sentences)
        
        print(f"✓ Extracted {len(all_sentences)} sentences")
        return all_sentences
    
    def _split_into_sentences(self, text: str, entity_spans: Dict) -> List[Dict]:
        """
        Split text into sentences and assign IOB2 labels
        """
        sentences = []
        
        # Simple sentence splitting (split on . ! ?)
        import re
        sent_boundaries = [0]
        for match in re.finditer(r'[.!?]\s+', text):
            sent_boundaries.append(match.end())
        sent_boundaries.append(len(text))
        
        for i in range(len(sent_boundaries) - 1):
            sent_start = sent_boundaries[i]
            sent_end = sent_boundaries[i + 1]
            sent_text = text[sent_start:sent_end].strip()
            
            if len(sent_text) < 3:
                continue
            
            # Tokenize sentence (simple whitespace tokenization)
            tokens = []
            labels = []
            
            # Track character positions
            char_pos = sent_start
            for word in sent_text.split():
                # Find word position in original text
                word_start = text.find(word, char_pos)
                word_end = word_start + len(word)
                
                # Check if word overlaps with any entity
                label = self._get_label(word_start, word_end, entity_spans)
                
                tokens.append(word)
                labels.append(label)
                
                char_pos = word_end
            
            if len(tokens) >= 3:  # Only keep sentences with at least 3 tokens
                sentences.append({
                    'tokens': tokens,
                    'labels': labels
                })
        
        return sentences
    
    def _get_label(self, word_start: int, word_end: int, entity_spans: Dict) -> str:
        """
        Get IOB2 label for a word based on entity spans
        """
        # Check if word overlaps with any entity
        for (ent_start, ent_end), entity_type in entity_spans.items():
            # Word starts within entity
            if ent_start <= word_start < ent_end:
                if word_start == ent_start:
                    return f"B-{entity_type}"
                else:
                    return f"I-{entity_type}"
            # Word overlaps entity
            elif word_start < ent_start < word_end:
                return f"B-{entity_type}"
        
        return "O"
    
    def save_conll(self, data: List[Dict], filepath: Path):
        """Save data in CoNLL format (2-column: token label)"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                tokens = item["tokens"]
                labels = item["labels"]
                
                for token, label in zip(tokens, labels):
                    f.write(f"{token} {label}\n")
                
                f.write("\n")  # Blank line between sentences
        
        print(f"✓ Saved {len(data)} sentences to {filepath.name}")
    
    def print_label_stats(self, data: List[Dict], split_name: str):
        """Print label distribution statistics"""
        label_counts = {}
        entity_counts = {}  # Count actual entities (B- tags only)
        total_tokens = 0
        
        for item in data:
            for label in item["labels"]:
                label_counts[label] = label_counts.get(label, 0) + 1
                total_tokens += 1
                
                # Count entities (B- tags)
                if label.startswith("B-"):
                    entity_type = label[2:]  # Remove B- prefix
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        print(f"\n{split_name} Statistics:")
        print(f"  Sentences: {len(data)}")
        print(f"  Tokens: {total_tokens}")
        print(f"  Total entities: {sum(entity_counts.values())}")
        
        if entity_counts:
            print(f"\n  Entity Type Distribution (entity mentions):")
            for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {entity_type:20s}: {count:7d} mentions")
        
        print(f"\n  Label Distribution (token level):")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            percentage = (count / total_tokens) * 100
            print(f"    {label:20s}: {count:7d} ({percentage:5.2f}%)")
    
    def split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test"""
        random.seed(MODEL_CONFIG["seed"])
        random.shuffle(data)
        
        total = len(data)
        train_size = int(total * TRAIN_RATIO)
        val_size = int(total * VAL_RATIO)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def update_config_labels(self):
        """Display detected labels for config update"""
        # Sort labels: O first, then B- labels, then I- labels
        sorted_labels = ["O"] if "O" in self.detected_labels else []
        b_labels = sorted([l for l in self.detected_labels if l.startswith("B-")])
        i_labels = sorted([l for l in self.detected_labels if l.startswith("I-")])
        
        sorted_labels.extend(b_labels)
        sorted_labels.extend(i_labels)
        
        print(f"\n" + "="*80)
        print("⚠️  UPDATE CONFIG FILE")
        print("="*80)
        print(f"\nDetected {len(sorted_labels)} unique labels in your data:")
        for i, label in enumerate(sorted_labels, 1):
            print(f"  {i:2d}. {label}")
        
        print(f"\n\nCopy this to training/config.py (replace NER_LABELS):\n")
        print("NER_LABELS = [")
        for label in sorted_labels:
            print(f'    "{label}",')
        print("]")
        print("\n" + "="*80)
    
    def run(self):
        """Run preprocessing pipeline"""
        print("="*80)
        print("LABEL STUDIO ANNOTATION PREPROCESSING")
        print("="*80)
        
        # Look for total.json file
        json_file = RAW_DATA_DIR / "total.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"File not found: {json_file}")
        
        # Load Label Studio annotations
        print("\n[1/5] Loading Label Studio annotations...")
        articles = self.load_label_studio_json(json_file)
        
        if not articles:
            raise ValueError(f"No articles loaded from {json_file}")
        
        # Convert to token-level format
        print("\n[2/5] Converting to token-level annotations...")
        all_sentences = self.convert_to_tokens(articles)
        
        if not all_sentences:
            raise ValueError("No sentences extracted from articles")
        
        print(f"\n✓ Total sentences: {len(all_sentences)}")
        print(f"✓ Detected {len(self.detected_labels)} unique labels")
        
        # Split data
        print("\n[3/5] Splitting data...")
        train_data, val_data, test_data = self.split_data(all_sentences)
        
        print(f"\n✓ Data split:")
        print(f"  Train: {len(train_data)} sentences ({TRAIN_RATIO*100:.0f}%)")
        print(f"  Val:   {len(val_data)} sentences ({VAL_RATIO*100:.0f}%)")
        print(f"  Test:  {len(test_data)} sentences ({TEST_RATIO*100:.0f}%)")
        
        # Create output directory
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        print("\n[4/5] Saving splits...")
        self.save_conll(train_data, PROCESSED_DATA_DIR / "train.conll")
        self.save_conll(val_data, PROCESSED_DATA_DIR / "val.conll")
        self.save_conll(test_data, PROCESSED_DATA_DIR / "test.conll")
        
        # Print statistics
        print("\n[5/5] Statistics:")
        self.print_label_stats(train_data, "Train")
        self.print_label_stats(val_data, "Validation")
        self.print_label_stats(test_data, "Test")
        
        # Show config update
        self.update_config_labels()
        
        print("\n" + "="*80)
        print("✅ PREPROCESSING COMPLETE")
        print("="*80)
        print(f"\nOutput saved to: {PROCESSED_DATA_DIR}")
        print(f"  - train.conll ({len(train_data)} sentences)")
        print(f"  - val.conll ({len(val_data)} sentences)")
        print(f"  - test.conll ({len(test_data)} sentences)")
        print(f"\n⚠️  NEXT STEP: Update training/config.py with the labels shown above")
        print(f"⚠️  Then run: python train.py --step train")


if __name__ == "__main__":
    preprocessor = FinanceDataPreprocessor()
    preprocessor.run()
