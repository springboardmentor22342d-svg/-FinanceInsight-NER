"""
PDF Extractor
Extract text and tables from financial PDFs using pdfplumber
"""

import pdfplumber
from pathlib import Path
from typing import Dict, List


class PDFExtractor:
    """
    Extract structured content from PDF files
    Uses pdfplumber for better table preservation
    """
    
    def __init__(self):
        pass
    
    def extract(self, pdf_path: str) -> Dict:
        """
        Extract text and tables from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            {
                "text": "Full extracted text",
                "pages": [{"page_num": 1, "text": "...", "tables": [...]}],
                "tables": [{"page": 1, "header": [...], "rows": [...]}]
            }
        """
        result = {
            "filename": Path(pdf_path).name,
            "text": "",
            "pages": [],
            "tables": []
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            full_text = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
                
                # Extract tables
                tables = page.extract_tables()
                page_tables = []
                
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    
                    # Structure table
                    structured_table = self._structure_table(table, page_num, table_idx)
                    if structured_table:
                        result["tables"].append(structured_table)
                        page_tables.append(structured_table)
                
                # Add page info
                result["pages"].append({
                    "page_num": page_num,
                    "text": page_text,
                    "table_count": len(page_tables)
                })
            
            result["text"] = "\n\n".join(full_text)
        
        return result
    
    def _structure_table(self, table: List[List[str]], page_num: int, table_idx: int) -> Dict:
        """
        Convert raw table to structured format
        
        Args:
            table: List of lists (rows)
            
        Returns:
            {
                "page": 1,
                "header": ["Item", "2024", "2023"],
                "rows": [{"item": "Revenue", "values": ["$391B", "$383B"]}]
            }
        """
        if not table or len(table) < 2:
            return None
        
        # First row is header
        header = [cell.strip() if cell else "" for cell in table[0]]
        
        # Process data rows
        rows = []
        for row in table[1:]:
            if not row or not row[0]:
                continue
            
            item = row[0].strip() if row[0] else ""
            values = [cell.strip() if cell else "" for cell in row[1:]]
            
            # Only include rows with actual data
            if item and any(values):
                rows.append({
                    "item": item,
                    "values": values
                })
        
        if not rows:
            return None
        
        return {
            "page": page_num,
            "index": table_idx,
            "header": header,
            "rows": rows
        }


if __name__ == "__main__":
    # Test
    extractor = PDFExtractor()
    result = extractor.extract("data/sample_pdfs/sample_10k.pdf")
    
    print(f"Extracted {len(result['pages'])} pages")
    print(f"Found {len(result['tables'])} tables")
