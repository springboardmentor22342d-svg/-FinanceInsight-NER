"""
Training Module
FinBERT NER model training pipeline
"""

# Only export what's needed, avoid circular imports
from .config import (
    MODEL_CONFIG,
    LABEL2ID,
    ID2LABEL,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)

# Don't import classes here to avoid circular imports
# Import them directly where needed

__all__ = [
    "MODEL_CONFIG",
    "LABEL2ID",
    "ID2LABEL",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODEL_DIR",
]
