import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
db = DocBin().from_disk("data/train/train.spacy")
docs = list(db.get_docs(nlp.vocab))

entity_count = sum(len(doc.ents) for doc in docs)
print("Total entities:", entity_count)
