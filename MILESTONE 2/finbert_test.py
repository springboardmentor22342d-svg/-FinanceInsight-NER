from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("finbert_ner_model")
model = AutoModelForTokenClassification.from_pretrained("finbert_ner_model")

ner = pipeline("ner", model=model, tokenizer=tokenizer)

text = "Microsoft reported revenue of $28B in Q1 2024."

results = ner(text)

for r in results:
    print(r)
