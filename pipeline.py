"""
Main Extraction Pipeline
Orchestrate complete extraction from PDF to insights.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from src.extractors import (
    PDFExtractor,
    SectionDetector,
    NERExtractor,
    TableParser,
    InsightGenerator
)


class FinancialInsightPipeline:
    """
    End-to-end pipeline for financial insight extraction
    """
    
    def __init__(self, model_path: str = None):
        """Initialize all extractors"""
        print("Initializing pipeline...")
        
        # Use config if model_path not provided
        if model_path is None:
            model_path = config['model']['model_path']
        
        self.pdf_extractor = PDFExtractor()
        self.section_detector = SectionDetector()
        self.ner_extractor = NERExtractor(model_path)
        self.table_parser = TableParser()
        self.insight_generator = InsightGenerator()
        
        # Load extraction config
        # self.extraction_config = config['extraction']
        
        print("✓ Pipeline ready")
    
    def extract(self, pdf_path: str, output_path: str = None) -> dict:
        """
        Run complete extraction pipeline
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output JSON path (optional)
            
        Returns:
            Extracted insights dict
        """
        print("\n" + "="*80)
        print("FINANCIAL INSIGHT EXTRACTION PIPELINE")
        print("="*80)
        print(f"\nInput: {pdf_path}")
        
        # Step 1: Extract PDF
        print("\n[1/5] Extracting PDF...")
        pdf_data = self.pdf_extractor.extract(pdf_path)
        print(f"  ✓ Extracted {len(pdf_data['pages'])} pages")
        print(f"  ✓ Found {len(pdf_data['tables'])} tables")
        
        # Step 2: Detect sections
        print("\n[2/5] Detecting sections...")
        sections = self.section_detector.detect_sections(pdf_data["text"])
        print(f"  ✓ Detected {len(sections)} sections:")
        for section_name in sections:
            print(f"      - {section_name}")
        
        # Step 3: Extract entities
        print("\n[3/5] Extracting entities...")
        section_entities = self.ner_extractor.extract_from_sections(sections)
        total_entities = sum(len(ents) for ents in section_entities.values())
        print(f"  ✓ Extracted {total_entities} entities")
        
        # Step 4: Parse tables
        print("\n[4/5] Parsing tables...")
        parsed_tables = self.table_parser.parse_tables(pdf_data["tables"])
        print(f"  ✓ Parsed {len(parsed_tables)} tables")
        
        # Step 5: Generate insights
        print("\n[5/5] Generating insights...")
        insights = self.insight_generator.generate(
            pdf_filename=Path(pdf_path).name,
            sections=sections,
            section_entities=section_entities,
            tables=parsed_tables
        )
        print(f"  ✓ Generated insights")
        
        # Create output folder if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save output
        if output_path is None:
            filename = Path(pdf_path).stem + "_insights.json"
            output_path = output_dir / filename
        else:
            # If output_path provided, save in output folder with that name
            output_path = output_dir / Path(output_path).name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved to: {output_path}")
        
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        
        # Print summary
        print("\nSummary:")
        print(f"  Company:      {insights['document_info']['company']}")
        print(f"  Fiscal Year:  {insights['document_info']['fiscal_year']}")
        print(f"  Currency:     {insights['document_info']['currency']}")
        print(f"  Sections:     {insights['metadata']['total_sections']}")
        print(f"  Entities:     {insights['metadata']['entities_extracted']}")
        print(f"  Tables:       {insights['metadata']['tables_parsed']}")
        
        return insights


def main():
    parser = argparse.ArgumentParser(
        description="Extract financial insights from PDF reports"
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to PDF file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON filename (will be saved in 'output' folder; default: <pdf_name>_insights.json)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/finbert_ner",
        help="Path to trained NER model (default: models/finbert_ner)"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = FinancialInsightPipeline(model_path=args.model)
    pipeline.extract(pdf_path=args.pdf, output_path=args.output)


if __name__ == "__main__":
    main()