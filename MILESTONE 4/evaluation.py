from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(true, pred):
    return {
        "Precision": precision_score(true, pred, average="weighted"),
        "Recall": recall_score(true, pred, average="weighted"),
        "F1 Score": f1_score(true, pred, average="weighted")
    }
