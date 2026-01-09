
"""
Quick sanity test for the trained NER model.
"""

import spacy

MODEL_PATH = "trained_model"

def main():
    nlp = spacy.load(MODEL_PATH)
    sample_text = "Infosys reported a net profit of Rs 1,619 crore in Q4."
    doc = nlp(sample_text)

    for ent in doc.ents:
        print(ent.text, ent.label_)


if __name__ == "__main__":
    main()
