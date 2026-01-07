import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("financial_news_events.csv")   # change filename if needed

print("\n===== HEAD OF DATA =====")
print(df.head())

print("\n===== BASIC INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ---------------------------------------------------------
# AUTO-DETECT TEXT COLUMN (avoids KeyError)
# ---------------------------------------------------------
possible_text_cols = ["text", "content", "headline", "news", "body", "article", "description"]

text_col = None
for col in df.columns:
    if col.lower() in possible_text_cols:
        text_col = col
        break

# If still not found, pick the longest text-like column
if text_col is None:
    text_col = df.select_dtypes(include="object").columns[0]

print(f"\n>>> Using text column: {text_col}\n")

# ---------------------------------------------------------
# TEXT LENGTH
# ---------------------------------------------------------
df["text_length"] = df[text_col].astype(str).apply(len)
print("\n===== TEXT LENGTH STATS =====")
print(df["text_length"].describe())

# ---------------------------------------------------------
# WORD COUNT
# ---------------------------------------------------------
df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
print("\n===== WORD COUNT STATS =====")
print(df["word_count"].describe())

# ---------------------------------------------------------
# MOST COMMON WORDS
# ---------------------------------------------------------
all_words = " ".join(df[text_col].astype(str)).lower()
words = re.findall(r"[a-zA-Z]+", all_words)

most_common_words = Counter(words).most_common(20)
print("\n===== TOP 20 WORDS =====")
for word, count in most_common_words:
    print(f"{word}: {count}")

# ---------------------------------------------------------
# PLOT TEXT LENGTH DISTRIBUTION
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["text_length"], bins=40)
plt.title("Text Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

# ---------------------------------------------------------
# PLOT WORD COUNT DISTRIBUTION
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["word_count"], bins=40)
plt.title("Word Count Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

