"""
Section Detector
Detect and segment financial document sections
"""

import re
from typing import Dict, List


class SectionDetector:
    """
    Detect sections in financial documents
    MD&A, Financial Statements, Risk Factors, etc.
    """
    
    # Known section headers (regex patterns)
    SECTION_PATTERNS = {
        "MD&A": [
            r"management'?s discussion and analysis",
            r"md\s*&\s*a",
            r"management discussion"
        ],
        "Financial Statements": [
            r"consolidated financial statements",
            r"financial statements",
            r"consolidated balance sheet",
            r"consolidated statements of operations"
        ],
        "Risk Factors": [
            r"risk factors",
            r"risks? and uncertainties"
        ],
        "Business Overview": [
            r"business overview",
            r"overview of business",
            r"item 1\.\s*business"
        ],
        "Results of Operations": [
            r"results of operations",
            r"operating results"
        ]
    }
    
    def __init__(self):
        pass
    
    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Segment text into sections
        
        Args:
            text: Full document text
            
        Returns:
            {
                "MD&A": "text of MD&A section...",
                "Financial Statements": "text of financial statements...",
                ...
            }
        """
        sections = {}
        lines = text.split('\n')
        
        current_section = "Unknown"
        current_content = []
        
        for line in lines:
            # Check if line is a section header
            detected_section = self._detect_section_header(line)
            
            if detected_section:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = detected_section
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _detect_section_header(self, line: str) -> str:
        """
        Check if line is a section header
        
        Returns:
            Section name if detected, None otherwise
        """
        line_lower = line.lower().strip()
        
        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    return section_name
        
        return None


if __name__ == "__main__":
    # Test
    detector = SectionDetector()
    
    test_text = """
    Management's Discussion and Analysis
    
    Our revenue increased to $391 billion...
    
    Consolidated Financial Statements
    
    Balance Sheet as of September 30, 2024...
    """
    
    sections = detector.detect_sections(test_text)
    print(f"Detected {len(sections)} sections:")
    for name in sections:
        print(f"  - {name}")
