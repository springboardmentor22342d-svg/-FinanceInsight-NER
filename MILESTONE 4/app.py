import json
import os

output = {
  "company": "Apple",
  "sections": {
    "MD&A": {
      "entities": [
        {
          "company": "Apple",
          "metric": "ORG",
          "value": "Management",
          "section": "MD&A"
        },
        {
          "company": "Apple",
          "metric": "MISC",
          "value": "1",
          "section": "MD&A"
        }
      ]
    }
  }
}

os.makedirs("outputs", exist_ok=True)

with open("outputs/final_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
