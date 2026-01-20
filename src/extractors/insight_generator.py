"""
Insight Generator
Generate final insights.json from extracted data
"""

import re
from datetime import datetime
from typing import Dict, List


class InsightGenerator:
    """
    Generate structured insights from extracted data
    """
    
    def __init__(self):
        pass
    
    def generate(
        self,
        pdf_filename: str,
        sections: Dict[str, str],
        section_entities: Dict[str, List[Dict]],
        tables: List[Dict]
    ) -> Dict:
        """
        Generate final insights JSON
        
        Args:
            pdf_filename: Original PDF filename
            sections: Detected sections
            section_entities: Extracted entities per section
            tables: Parsed tables
            
        Returns:
            Complete insights structure
        """
        insights = {
            "document_info": self._extract_document_info(
                pdf_filename, sections, section_entities
            ),
            "narrative_insights": self._extract_narrative_insights(
                section_entities, sections
            ),
            "financial_tables": tables,
            "metadata": {
                "total_sections": len(sections),
                "entities_extracted": sum(len(ents) for ents in section_entities.values()),
                "tables_parsed": len(tables),
                "extracted_at": datetime.now().isoformat()
            }
        }
        
        return insights
    
    def _extract_document_info(
        self,
        filename: str,
        sections: Dict[str, str],
        section_entities: Dict[str, List[Dict]]
    ) -> Dict:
        """Extract high-level document metadata"""
        # Extract company name
        company = self._find_company_name(section_entities)
        
        # Extract fiscal year
        fiscal_year = self._find_fiscal_year(section_entities, sections)
        
        # Extract currency unit
        currency = self._detect_currency(section_entities)
        
        return {
            "filename": filename,
            "company": company,
            "fiscal_year": fiscal_year,
            "currency": currency,
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _find_company_name(self, section_entities: Dict[str, List[Dict]]) -> str:
        """Find company name from ORG entities"""
        # Priority sections
        priority_sections = ["Business Overview", "MD&A", "Unknown"]
        
        for section in priority_sections:
            if section in section_entities:
                for entity in section_entities[section]:
                    if entity["label"] == "ORG":
                        return entity["text"]
        
        # Fallback: any ORG entity
        for entities in section_entities.values():
            for entity in entities:
                if entity["label"] == "ORG":
                    return entity["text"]
        
        return "Unknown"
    
    def _find_fiscal_year(
        self,
        section_entities: Dict[str, List[Dict]],
        sections: Dict[str, str]
    ) -> str:
        """Find fiscal year from DATE entities or text"""
        # Try DATE entities first
        for entities in section_entities.values():
            for entity in entities:
                if entity["label"] == "DATE":
                    year_match = re.search(r'20\d{2}', entity["text"])
                    if year_match:
                        return year_match.group(0)
        
        # Search in text
        for text in sections.values():
            match = re.search(r'fiscal year\s*(20\d{2})', text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def _detect_currency(self, section_entities: Dict[str, List[Dict]]) -> str:
        """Detect currency from MONEY entities"""
        for entities in section_entities.values():
            for entity in entities:
                if entity["label"] == "MONEY":
                    text = entity["text"]
                    
                    if "Rs" in text or "₹" in text:
                        return "INR"
                    elif "$" in text or "USD" in text:
                        return "USD"
                    elif "€" in text or "EUR" in text:
                        return "EUR"
        
        return "USD"  # Default
    
    def _extract_narrative_insights(
        self,
        section_entities: Dict[str, List[Dict]],
        sections: Dict[str, str]
    ) -> List[Dict]:
        """
        Extract narrative insights from entities
        
        Returns:
            [
                {
                    "section": "MD&A",
                    "company": "Apple",
                    "metric": "Revenue",
                    "value": "$391,035 million",
                    "period": "2024"
                }
            ]
        """
        insights = []
        
        for section_name, entities in section_entities.items():
            section_text = sections.get(section_name, "")
            
            # Find company in this section
            company = None
            for entity in entities:
                if entity["label"] == "ORG":
                    company = entity["text"]
                    break
            
            # Find period
            period = None
            for entity in entities:
                if entity["label"] == "DATE":
                    year_match = re.search(r'20\d{2}', entity["text"])
                    if year_match:
                        period = year_match.group(0)
                        break
            
            # Extract metrics from MONEY entities
            for entity in entities:
                if entity["label"] == "MONEY":
                    metric = self._determine_metric(entity, section_text)
                    
                    if metric:
                        insights.append({
                            "section": section_name,
                            "company": company,
                            "metric": metric,
                            "value": entity["text"],
                            "period": period
                        })
        
        return insights
    
    def _determine_metric(self, money_entity: Dict, context: str) -> str:
        """
        Determine what metric a MONEY entity refers to
        by checking surrounding context
        """
        start = money_entity.get("start", 0)
        
        # Get context window (100 chars before)
        context_start = max(0, start - 100)
        context_window = context[context_start:start + 50].lower()
        
        # Check for keywords
        if "revenue" in context_window or "sales" in context_window:
            return "Revenue"
        elif "profit" in context_window:
            return "Net Profit"
        elif "income" in context_window:
            return "Net Income"
        elif "assets" in context_window:
            return "Total Assets"
        elif "liabilities" in context_window:
            return "Total Liabilities"
        elif "equity" in context_window:
            return "Shareholders' Equity"
        
        return None


if __name__ == "__main__":
    # Test
    generator = InsightGenerator()
    
    test_entities = {
        "MD&A": [
            {"text": "Apple Inc", "label": "ORG", "start": 0, "end": 9},
            {"text": "$391 billion", "label": "MONEY", "start": 30, "end": 42},
            {"text": "2024", "label": "DATE", "start": 46, "end": 50}
        ]
    }
    
    test_sections = {
        "MD&A": "Apple Inc reported revenue of $391 billion in 2024."
    }
    
    info = generator._extract_document_info("test.pdf", test_sections, test_entities)
    print("Document info:", info)
