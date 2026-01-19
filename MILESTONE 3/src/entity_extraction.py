import os
import json
import re

# -----------------------------
# Step 1: Load entity configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
config_path = os.path.join(BASE_DIR, "../config/entities.json")

with open(config_path, "r") as f:
    entity_keywords = json.load(f)

# -----------------------------
# Step 2: User input - select entities to extract
# -----------------------------
# Example: user wants to extract EPS and Revenue Growth
user_entities = ["EPS", "REVENUE_GROWTH"]

# -----------------------------
# Step 3: Sample financial text
# -----------------------------
text = "Apple reported EPS of $3.25 and revenue grew by 12% this quarter."
text = "Microsoft reported EPS of $2.50, revenue grew by 10%, and net income was $5B."


# -----------------------------
# Step 4: Regex-based extraction
# -----------------------------
filtered_results = []

# EPS extraction
if "EPS" in user_entities:
    eps_pattern = r"(EPS\s*(?:of\s*)?\$?\d+(?:\.\d+)?)"
    eps_matches = re.findall(eps_pattern, text, re.IGNORECASE)
    if eps_matches:
        filtered_results.append({"EPS": eps_matches[0]})

# Revenue Growth extraction
if "REVENUE_GROWTH" in user_entities:
    rev_pattern = r"((?:revenue|sales)\s+grew\s+by\s+\d+%)"
    rev_matches = re.findall(rev_pattern, text, re.IGNORECASE)
    if rev_matches:
        filtered_results.append({"REVENUE_GROWTH": rev_matches[0]})

# -----------------------------
# Step 5: Print final extracted entities
# -----------------------------
print("Filtered Entities:", filtered_results)

# -----------------------------
# Step 6: Save results to outputs folder
# -----------------------------
output_path = os.path.join(BASE_DIR, "../outputs/entities_output.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(filtered_results, f, indent=4)

print(f"Results saved to {output_path}")
