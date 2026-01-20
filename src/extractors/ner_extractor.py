"""
NER Extractor
Load trained FinBERT model and extract entities from text
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict


class NERExtractor:
    """
    Extract named entities using trained FinBERT NER model
    """
    
    def __init__(self, model_path: str = "models/finbert_ner"):
        """
        Load trained model
        
        Args:
            model_path: Path to trained model directory
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please train the model first: python train.py"
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mapping
        self.id2label = self.model.config.id2label
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text
        
        Args:
            text: Input text
            
        Returns:
            [
                {"text": "Apple Inc", "label": "ORG", "start": 0, "end": 9},
                {"text": "$394B", "label": "MONEY", "start": 30, "end": 35}
            ]
        """
        if not text or len(text.strip()) < 10:
            return []
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[p.item()] for p in predictions[0]]
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()
        
        # Extract entities
        entities = []
        current_entity = None
        
        for token, label, (start, end) in zip(tokens, predicted_labels, offset_mapping):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>'] or start == end == 0:
                continue
            
            # Convert numpy types to Python int
            start = int(start)
            end = int(end)
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": start,
                    "end": end
                }
            
            elif label.startswith('I-'):
                # Continue current entity
                if current_entity:
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"]:end]
            
            else:  # O tag
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Save last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def extract_from_sections(self, sections: Dict[str, str]) -> Dict[str, List[Dict]]:
        """
        Extract entities from multiple sections
        
        Args:
            sections: {section_name: text}
            
        Returns:
            {section_name: [entities]}
        """
        section_entities = {}
        
        for section_name, text in sections.items():
            entities = self.extract_entities(text)
            if entities:
                section_entities[section_name] = entities
        
        return section_entities


if __name__ == "__main__":
    # Test
    extractor = NERExtractor()
    
    text = "Apple Inc reported revenue of $394 billion in fiscal year 2024."
    entities = extractor.extract_entities(text)
    
    print("Extracted entities:")
    for ent in entities:
        print(f"  {ent['text']:20s} [{ent['label']}]")
