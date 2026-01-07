import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nlp = spacy.load("en_core_web_sm")

# Uncomment only the first time:
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")

FINANCE_TERMS = {
    r"\bebitda\b": "earnings_before_interest_taxes_depreciation_and_amortization",
    r"\bp\/e\b": "price_to_earnings_ratio",
    r"\byoy\b": "year_over_year",
}

CURRENCY_MAP = {
    "$": "usd",
    "₹": "inr",
}

def preprocess_finance_text(text):

    text = text.lower()

    for symbol, name in CURRENCY_MAP.items():
        text = text.replace(symbol, f" {name} ")

    for pattern, replacement in FINANCE_TERMS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"(\d+)\s?m\b", r"\1 million", text)
    text = re.sub(r"(\d+)\s?b\b", r"\1 billion", text)

    text = re.sub(r"[^a-zA-Z0-9.% ]+", " ", text)

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]

    return {
        "clean_text": text,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmas
    }

if __name__ == "__main__":
    sample = """
    Infosys reported EBITDA of ₹450M in FY24, while P/E ratio dropped to 18.5.
    Revenue grew 12% YoY reaching $5.7B.
    """

    result = preprocess_finance_text(sample)

    print("\nCleaned Text:", result["clean_text"])
    print("\nTokens:", result["tokens"])
    print("\nPOS Tags:", result["pos_tags"])
    print("\nLemmas:", result["lemmas"])



