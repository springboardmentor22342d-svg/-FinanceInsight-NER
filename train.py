"""
Training Entry Point
Run complete training pipeline from command line
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.preprocess import FinanceDataPreprocessor
from training.train_ner import FinBERTNERTrainer
from training.evaluate import NEREvaluator


def main():
    parser = argparse.ArgumentParser(description="Train FinBERT NER Model")
    
    parser.add_argument(
        "--step",
        type=str,
        choices=["preprocess", "train", "evaluate", "all"],
        default="all",
        help="Which step to run (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.step in ["preprocess", "all"]:
        print("\n" + "="*80)
        print("STEP 1: PREPROCESSING")
        print("="*80)
        preprocessor = FinanceDataPreprocessor()
        preprocessor.run()  # ← No arguments needed
    
    if args.step in ["train", "all"]:
        print("\n" + "="*80)
        print("STEP 2: TRAINING")
        print("="*80)
        trainer = FinBERTNERTrainer()
        trainer.train()
    
    if args.step in ["evaluate", "all"]:
        print("\n" + "="*80)
        print("STEP 3: EVALUATION")
        print("="*80)
        evaluator = NEREvaluator()
        evaluator.evaluate()
    
    print("\n" + "="*80)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
