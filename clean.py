import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ------------------------------------
# 1. DOWNLOAD NLTK MODELS (Python 3.12+)
# ------------------------------------
nltk.download("punkt")        # new NLTK tokenizer
nltk.download("punkt_tab")    # fix for Python 3.12/3.13
nltk.download("stopwords")

# ------------------------------------
# 2. LOAD SPACY MODEL
# ------------------------------------
nlp = spacy.load("en_core_web_sm")

# ------------------------------------
# 3. CLEANING FUNCTION
# ------------------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9$€%\.\-/ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------
# 4. TOKENIZATION
# ------------------------------------
def tokenize(text):
    return word_tokenize(text)

# ------------------------------------
# 5. STOPWORDS
# ------------------------------------
stop_words = set(stopwords.words("english"))

def remove_stopwords(tokens):
    return [w for w in tokens if w not in stop_words]

# ------------------------------------
# 6. LEMMATIZATION
# ------------------------------------
def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

# ------------------------------------
# 7. MAIN EXECUTION
# ------------------------------------
df = pd.read_csv("Finance_data.csv")

print(df.columns)

df["clean_text"] = df["Investment_Avenues"].apply(clean_text)
df["tokens"] = df["clean_text"].apply(tokenize)
df["no_stopwords"] = df["tokens"].apply(remove_stopwords)
df["lemmatized"] = df["no_stopwords"].apply(lemmatize)

print(df.head())




