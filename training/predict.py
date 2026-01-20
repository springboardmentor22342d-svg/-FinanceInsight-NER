"""
Simple NER Inference - Extract entities from financial text
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

class FinancialNER:
    def __init__(self, model_path: str = "models/finbert_ner"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        print("✓ Model ready")
    
    def extract_entities(self, text: str):
        """Extract entities from text"""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Extract entities
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = [self.id2label[p.item()] for p in predictions[0]]
        
        entities = []
        current_entity = None
        
        for token, label in zip(tokens, labels):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            if label.startswith("B-"):
                # Save previous
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                current_entity = {
                    "text": token.replace("##", ""),
                    "type": label[2:]
                }
            
            elif label.startswith("I-") and current_entity:
                # Continue entity
                current_entity["text"] += token.replace("##", "")
            
            else:
                # End entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities


# Usage example
if __name__ == "__main__":
    ner = FinancialNER()
    
    # Example text
    text = """
    Apple Inc reported revenue of $95 billion in Q4 2024, up 12 percent from last year.
    CEO Tim Cook announced the results on January 15, 2025 in Cupertino.
    """
    
    entities = ner.extract_entities(text)
    print(text)
    
    print("\nExtracted Entities:")
    print("-" * 50)
    for entity in entities:
        print(f"{entity['text']:30s} → {entity['type']}")
