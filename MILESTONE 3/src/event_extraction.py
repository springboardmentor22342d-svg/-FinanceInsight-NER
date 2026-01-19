import os
import json
import re
from datetime import datetime

# -----------------------------
# Step 1: Load event configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "../config/events.json")

with open(config_path, "r") as f:
    event_keywords = json.load(f)

# -----------------------------
# Step 2: Sample financial text
# -----------------------------
text = """
Apple announced a stock split of 4-for-1 on 08/31/2025.
Tesla held its earnings call on 12/28/2025.
Microsoft acquired Nuance Communications on 11/10/2025.
Airbnb is planning its IPO on 03/15/2026.
"""


# -----------------------------
# Step 3: User-defined timeframe
# -----------------------------
start_date = "01/01/2000"
end_date = "12/31/2030"

def in_range(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%m/%d/%Y")
        start_obj = datetime.strptime(start_date, "%m/%d/%Y")
        end_obj = datetime.strptime(end_date, "%m/%d/%Y")
        return start_obj <= date_obj <= end_obj
    except:
        return False

# -----------------------------
# Step 4: Extract events
# -----------------------------
filtered_events = []

for event, keywords in event_keywords.items():
    for keyword in keywords:
        # improved regex for keyword + optional date
        pattern = re.compile(rf"({keyword}.*?)(?: on )?(\d{{2}}/\d{{2}}/\d{{4}})?[.,]?", re.IGNORECASE)
        matches = pattern.findall(text)
        for match in matches:
            event_text, date = match
            if date is None or in_range(date):
                filtered_events.append({
                    "event_type": event,
                    "text": event_text.strip(),
                    "date": date
                })

# -----------------------------
# Step 5: Print results
# -----------------------------
print("Filtered Financial Events:", filtered_events)

# -----------------------------
# Step 6: Save to outputs
# -----------------------------
output_path = os.path.join(BASE_DIR, "../outputs/events_output.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(filtered_events, f, indent=4)

print(f"Results saved to {output_path}")
