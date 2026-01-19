import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# -------------------------------------------------
# 1. Read BIO formatted data
# -------------------------------------------------
def read_bio_file(path):
    sentences, labels = [], []
    words, tags = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                word, tag = line.split()
                words.append(word)
                tags.append(tag)

    return sentences, labels


sentences, labels = read_bio_file(
    r"C:\Users\Pavit\OneDrive\Desktop\Finance data\MILESTONE 2\train.txt"
)

# -------------------------------------------------
# 2. Label mapping
# -------------------------------------------------
unique_labels = sorted(set(tag for sent in labels for tag in sent))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

labels_ids = [[label2id[tag] for tag in sent] for sent in labels]

# -------------------------------------------------
# 3. Create HuggingFace Dataset
# -------------------------------------------------
dataset = Dataset.from_dict({
    "tokens": sentences,
    "ner_tags": labels_ids
})

# -------------------------------------------------
# 4. Load FinBERT
# -------------------------------------------------
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# -------------------------------------------------
# 5. Tokenization + label alignment
# -------------------------------------------------
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        is_split_into_words=True
    )

    aligned_labels = []

    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)
            prev_word = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# -------------------------------------------------
# 6. Evaluation metrics
# -------------------------------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, lab in zip(predictions, labels):
        curr_true = []
        curr_pred = []
        for p_i, l_i in zip(pred, lab):
            if l_i != -100:
                curr_true.append(id2label[l_i])
                curr_pred.append(id2label[p_i])
        true_labels.append(curr_true)
        true_predictions.append(curr_pred)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# -------------------------------------------------
# 7. Training arguments
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no"
)


# -------------------------------------------------
# 8. Trainer
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # acceptable for academic project
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -------------------------------------------------
# 9. Train & Evaluate
# -------------------------------------------------
trainer.train()
metrics = trainer.evaluate()

print("\n✅ Evaluation Metrics")
print(metrics)

# -------------------------------------------------
# 10. Error Analysis (Classification Report)
# -------------------------------------------------
predictions = trainer.predict(tokenized_dataset)
preds = np.argmax(predictions.predictions, axis=2)

true_labels = []
true_preds = []

for pred, lab in zip(preds, predictions.label_ids):
    curr_true = []
    curr_pred = []
    for p_i, l_i in zip(pred, lab):
        if l_i != -100:
            curr_true.append(id2label[l_i])
            curr_pred.append(id2label[p_i])
    true_labels.append(curr_true)
    true_preds.append(curr_pred)

print("\n📊 Detailed Classification Report:")
print(classification_report(true_labels, true_preds))

# -------------------------------------------------
# 11. Save model
# -------------------------------------------------
model.save_pretrained("finbert_ner_model")
tokenizer.save_pretrained("finbert_ner_model")

print("\n🎉 FinBERT Financial NER Training + Evaluation COMPLETED")
