"""
CLI for Interactive NER Testing

Test entity extraction on sentences and PDF documents
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.ner_extractor import NERExtractor
from src.extractors.pdf_extractor import PDFExtractor


class FinanceNERCLI:
    def __init__(self, model_path: str = "models/finbert_ner"):
        """Initialize CLI with trained FinBERT NER model"""
        print(f"Loading model from {model_path}...")
        try:
            self.ner = NERExtractor(model_path)
            self.pdf_extractor = PDFExtractor()
            print("✓ Model loaded successfully\n")
        except FileNotFoundError as e:
            print(f"✗ Model not found at {model_path}")
            print("✗ Train the model first: python train.py")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def process_sentence(self, text: str) -> List[Dict]:
        """Extract entities from a single sentence"""
        if not text or len(text.strip()) < 10:
            print("⚠ Text too short (minimum 10 characters)")
            return []
        
        entities = self.ner.extract_entities(text)
        return entities
    
    def process_pdf(self, pdf_path: str) -> tuple[List[Dict], str, dict]:
        """
        Extract text from PDF and get entities
        Returns: (entities, full_text, pdf_metadata)
        """
        try:
            print(f"Extracting text from PDF: {pdf_path}")
            pdf_data = self.pdf_extractor.extract(pdf_path)
            
            full_text = pdf_data['text']
            print(f"✓ Extracted {len(pdf_data['pages'])} pages")
            print(f"✓ Found {len(pdf_data['tables'])} tables\n")
            
        except FileNotFoundError:
            print(f"✗ PDF file not found: {pdf_path}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error extracting PDF: {e}")
            sys.exit(1)
        
        if len(full_text.strip()) < 10:
            print("⚠ Extracted text too short (minimum 10 characters)")
            return [], full_text, pdf_data
        
        # Extract entities from full text
        print("Extracting entities...")
        entities = self.ner.extract_entities(full_text)
        print(f"✓ Extracted {len(entities)} entities\n")
        
        return entities, full_text, pdf_data
    
    def display_text_format(self, entities: List[Dict], source_text: str = None, show_context: bool = True):
        """Display entities in readable text format"""
        if not entities:
            print("No entities found.\n")
            return
        
        print("\n" + "="*80)
        print(f"EXTRACTED ENTITIES ({len(entities)} found)")
        print("="*80 + "\n")
        
        for i, ent in enumerate(entities, 1):
            print(f"{i}. \"{ent['text']}\"")
            print(f"   Type: {ent['label']}")
            print(f"   Position: [{ent['start']}:{ent['end']}]")
            
            # Show context if available and requested
            if show_context and source_text:
                start_ctx = max(0, ent['start'] - 40)
                end_ctx = min(len(source_text), ent['end'] + 40)
                context = source_text[start_ctx:end_ctx].replace('\n', ' ').strip()
                print(f"   Context: ...{context}...")
            print()
    
    def display_json_format(self, entities: List[Dict], pdf_metadata: dict = None):
        """Display entities in JSON format with optional PDF metadata"""
        output = {
            "total_entities": len(entities),
            "entities": entities
        }
        
        if pdf_metadata:
            output["pdf_metadata"] = {
                "filename": pdf_metadata.get('filename'),
                "total_pages": len(pdf_metadata.get('pages', [])),
                "total_tables": len(pdf_metadata.get('tables', []))
            }
        
        print(json.dumps(output, indent=2, ensure_ascii=False))
    
    def get_statistics(self, entities: List[Dict]):
        """Generate and display entity statistics"""
        if not entities:
            print("\nNo entities to analyze.\n")
            return
        
        # Count by label
        label_counts = {}
        for ent in entities:
            label = ent['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\n" + "="*80)
        print("ENTITY STATISTICS")
        print("="*80)
        print(f"\nTotal Entities: {len(entities)}")
        print(f"Unique Entity Types: {len(label_counts)}\n")
        
        print("Breakdown by Type:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(entities)) * 100
            bar = "█" * int(percentage / 2)  # Visual bar
            print(f"  {label:15s}: {count:4d} ({percentage:5.1f}%) {bar}")
        print()
    
    def save_output(self, entities: List[Dict], output_path: str, pdf_metadata: dict = None):
        """Save results to JSON file"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / output_path
        
        result = {
            "total_entities": len(entities),
            "entities": entities
        }
        
        if pdf_metadata:
            result["source_document"] = {
                "filename": pdf_metadata.get('filename'),
                "total_pages": len(pdf_metadata.get('pages', [])),
                "total_tables": len(pdf_metadata.get('tables', []))
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Financial NER CLI - Test entity extraction on sentences or PDF documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single sentence
  python cli.py --text "Apple Inc. reported revenue of $394.3 billion in fiscal year 2022."
  
  # Process a PDF annual report
  python cli.py --pdf data/pdfs/tesla_10k.pdf --stats
  
  # Get JSON output from PDF
  python cli.py --pdf data/pdfs/apple_annual_report.pdf --format json
  
  # Save PDF results with statistics
  python cli.py --pdf data/pdfs/microsoft_10q.pdf --output msft_entities.json --stats
  
  # Use custom model checkpoint
  python cli.py --model models/finbert_ner_v2 --pdf data/pdfs/amazon_earnings.pdf
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/finbert_ner',
        help='Path to trained FinBERT NER model (default: models/finbert_ner)'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--text',
        type=str,
        help='Input sentence to analyze'
    )
    group.add_argument(
        '--pdf',
        type=str,
        help='Path to PDF file (annual report, 10-K, earnings report, etc.)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show entity statistics'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file in output/ folder'
    )
    
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Hide context snippets in text output'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = FinanceNERCLI(args.model)
    
    # Process input
    pdf_metadata = None
    if args.text:
        print(f"Input: \"{args.text}\"\n")
        entities = cli.process_sentence(args.text)
        source_text = args.text
    else:
        entities, source_text, pdf_metadata = cli.process_pdf(args.pdf)
    
    # Display results
    if args.format == 'json':
        cli.display_json_format(entities, pdf_metadata)
    else:
        show_context = not args.no_context
        cli.display_text_format(entities, source_text, show_context)
    
    # Show statistics
    if args.stats:
        cli.get_statistics(entities)
    
    # Save output
    if args.output:
        cli.save_output(entities, args.output, pdf_metadata)
    
    print("="*80)
    print("✓ Complete")
    print("="*80)


if __name__ == '__main__':
    main()
