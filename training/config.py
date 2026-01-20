"""
Training Configuration
Standalone version - no external dependencies
"""

import os
from pathlib import Path

# Project root (training folder's parent)
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "finbert_ner"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "base_model": "yiyanghkust/finbert-tone",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 50,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "fp16": True
}

# NER Labels - customize based on your CoNLL files
NER_LABELS = [
    "O",
    "B-DATE",
    "B-EVENT",
    "B-GPE",
    "B-INSTRUMENT",
    "B-METRIC",
    "B-MONEY",
    "B-ORG",
    "B-PERCENT",
    "B-PERSON",
    "B-PRODUCT",
    "B-TICKER",
    "I-DATE",
    "I-EVENT",
    "I-GPE",
    "I-INSTRUMENT",
    "I-METRIC",
    "I-MONEY",
    "I-ORG",
    "I-PERCENT",
    "I-PERSON",
    "I-PRODUCT",
    "I-TICKER",
]

# Label mappings
LABEL2ID = {label: idx for idx, label in enumerate(NER_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Data Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Training Configuration
EARLY_STOPPING_PATIENCE = 3
LOGGING_STEPS = 100
EVAL_STEPS = 500
SAVE_STEPS = 500
