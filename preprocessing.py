import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# ----------------------------------
# DOWNLOAD ALL REQUIRED NLTK MODELS
# ----------------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

# ----------------------------------
# LOAD DATA
# ----------------------------------
df = pd.read_csv("Finance_data.csv")

# ----------------------------------
# CLEANING
# ----------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------
# TOKENIZE → STOPWORDS → POS TAG → LEMMA
# ----------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]

    pos = pos_tag(tokens)

    def get_wordnet_pos(tag):
        if tag.startswith("J"):
            return "a"
        elif tag.startswith("V"):
            return "v"
        elif tag.startswith("N"):
            return "n"
        elif tag.startswith("R"):
            return "r"
        else:
            return "n"

    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos]
    return " ".join(lemmatized)

# APPLY TO YOUR COLUMN
df["processed_text"] = df["Investment_Avenues"].apply(preprocess)

print(df.head())
print("DONE")
