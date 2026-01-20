"""
Table Parser
Parse extracted tables into clean multi-year format
"""

import re
from typing import List, Dict


class TableParser:
    """
    Parse financial tables into structured format
    """
    
    # Known table types
    TABLE_TYPES = {
        "Balance Sheet": ["balance sheet", "financial position"],
        "Income Statement": ["income statement", "operations", "earnings"],
        "Cash Flow": ["cash flow"],
        "Equity Statement": ["shareholders' equity", "stockholders' equity"]
    }
    
    def __init__(self):
        pass
    
    def parse_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Parse tables into clean format
        
        Args:
            tables: Raw tables from PDF extractor
            
        Returns:
            [
                {
                    "name": "Consolidated Balance Sheet",
                    "type": "Balance Sheet",
                    "rows": [
                        {"item": "Total Assets", "2024": "$364,980", "2023": "$352,755"}
                    ]
                }
            ]
        """
        parsed_tables = []
        
        for table in tables:
            parsed = self._parse_single_table(table)
            if parsed:
                parsed_tables.append(parsed)
        
        return parsed_tables
    
    def _parse_single_table(self, table: Dict) -> Dict:
        """Parse single table"""
        header = table.get("header", [])
        raw_rows = table.get("rows", [])
        
        if not header or not raw_rows:
            return None
        
        # Detect table name and type
        table_name = self._detect_table_name(header, raw_rows)
        table_type = self._classify_table(table_name)
        
        # Parse rows into year-based format
        parsed_rows = self._parse_rows(header, raw_rows)
        
        if not parsed_rows:
            return None
        
        return {
            "name": table_name,
            "type": table_type,
            "page": table.get("page"),
            "rows": parsed_rows
        }
    
    def _detect_table_name(self, header: List[str], rows: List[Dict]) -> str:
        """
        Detect table name from header or first rows
        """
        # Try header first
        header_text = " ".join(header).strip()
        if len(header_text) > 10:
            return header_text
        
        # Try first row if it looks like a title
        if rows and rows[0]:
            first_item = rows[0].get("item", "")
            if first_item and len(first_item) > 20 and not any(char.isdigit() for char in first_item):
                return first_item
        
        return "Financial Table"
    
    def _classify_table(self, table_name: str) -> str:
        """Classify table type"""
        table_name_lower = table_name.lower()
        
        for table_type, keywords in self.TABLE_TYPES.items():
            if any(kw in table_name_lower for kw in keywords):
                return table_type
        
        return "Other"
    
    def _parse_rows(self, header: List[str], raw_rows: List[Dict]) -> List[Dict]:
        """
        Parse rows into multi-year format
        
        Args:
            header: ["Item", "2024", "2023", "2022"]
            raw_rows: [{"item": "Revenue", "values": ["$391B", "$383B", "$394B"]}]
            
        Returns:
            [{"item": "Revenue", "2024": "$391B", "2023": "$383B", "2022": "$394B"}]
        """
        parsed_rows = []
        
        # Extract year labels from header
        year_labels = self._extract_year_labels(header)
        
        for row in raw_rows:
            item = row.get("item", "").strip()
            values = row.get("values", [])
            
            if not item or not values:
                continue
            
            # Create row dict
            parsed_row = {"item": item}
            
            # Map values to years
            for i, value in enumerate(values):
                if i < len(year_labels):
                    year = year_labels[i]
                    parsed_row[year] = value.strip()
            
            # Only include rows with at least one value
            if len(parsed_row) > 1:
                parsed_rows.append(parsed_row)
        
        return parsed_rows
    
    def _extract_year_labels(self, header: List[str]) -> List[str]:
        """
        Extract year labels from header
        
        Args:
            header: ["", "2024", "2023", "2022"]
            
        Returns:
            ["2024", "2023", "2022"]
        """
        years = []
        
        for col in header[1:]:  # Skip first column (usually "Item")
            # Look for year pattern (4 digits)
            year_match = re.search(r'(20\d{2})', col)
            if year_match:
                years.append(year_match.group(1))
            elif col.strip():
                # Use column as-is if not empty
                years.append(col.strip())
        
        # Fallback: generate generic column names
        if not years:
            years = [f"Col{i+1}" for i in range(len(header) - 1)]
        
        return years


if __name__ == "__main__":
    # Test
    parser = TableParser()
    
    test_table = {
        "header": ["", "2024", "2023"],
        "rows": [
            {"item": "Total Assets", "values": ["$364,980", "$352,755"]},
            {"item": "Total Liabilities", "values": ["$308,030", "$290,020"]}
        ]
    }
    
    parsed = parser._parse_single_table(test_table)
    print("Parsed table:")
    for row in parsed["rows"]:
        print(f"  {row}")
