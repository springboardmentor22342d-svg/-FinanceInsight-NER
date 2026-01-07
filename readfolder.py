import os
import pandas as pd

base_path = "dataset/"   # change this to your main folder

all_sentences = []

# Walk through all folders & files inside dataset/
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".txt"):     # reading .txt files
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                line = f.read().strip()
                if line:
                    all_sentences.append(line)

# Create DataFrame
df = pd.DataFrame({"text": all_sentences})

# Show output
print(df.head())
print("Total sentences loaded:", len(df))
