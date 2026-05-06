"""
train.py
---------
Main training script — runs both ML models, logs to MLflow, saves artifacts.

Usage:
    python train.py                    # generates data + trains both models
    python train.py --data-only        # only generate data
    python train.py --model1-only      # only train Model 1 (TF-IDF + RF)
    python train.py --model2-only      # only train Model 2 (BERT)
    python train.py --skip-generate    # use existing data/resume_dataset.csv
"""

import os
import sys
import argparse
import json
import pandas as pd
import mlflow

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import generate_dataset
from models.model1_tfidf_rf import TFIDFRandomForestScreener
from models.model2_bert import BERTSimilarityScreener


# ─── Configg ───────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = "mlruns"          # local folder (can change to remote URI)
DATA_PATH = "data/resume_dataset.csv"
MODEL1_SAVE_PATH = "saved_models/model1_tfidf_rf.pkl"
MODEL2_SAVE_PATH = "saved_models/model2_bert.pkl"
RESULTS_PATH = "training_results.json"
N_SAMPLES = 400


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.makedirs(MLFLOW_TRACKING_URI, exist_ok=True)
    print(f"[MLflow] Tracking URI: {os.path.abspath(MLFLOW_TRACKING_URI)}")
    print(f"[MLflow] View UI with: mlflow ui --port 5000")


def load_or_generate_data(skip_generate: bool = False) -> pd.DataFrame:
    """Load existing dataset or generate a new one."""
    if skip_generate and os.path.exists(DATA_PATH):
        print(f"[Data] Loading existing dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"[Data] Generating synthetic dataset ({N_SAMPLES} samples)...")
        df = generate_dataset(N_SAMPLES)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"[Data] Saved to {DATA_PATH}")

    print(f"[Data] Total: {len(df)} | Match: {df['label'].sum()} | No-Match: {(df['label']==0).sum()}")
    return df


def train_model1(df: pd.DataFrame) -> dict:
    """Train TF-IDF + Random Forest model."""
    print("\n" + "="*60)
    print("  MODEL 1: TF-IDF + Random Forest")
    print("="*60)

    model1 = TFIDFRandomForestScreener(
        n_estimators=200,
        max_depth=15,
        max_features_tfidf=8000,
        threshold=0.50,
    )
    metrics = model1.train(df)
    model1.save(MODEL1_SAVE_PATH)
    return {"model": "TF-IDF + Random Forest", **metrics}


def train_model2(df: pd.DataFrame) -> dict:
    """Train BERT Similarity model."""
    print("\n" + "="*60)
    print("  MODEL 2: BERT Similarity + Logistic Regression")
    print("="*60)

    model2 = BERTSimilarityScreener(
        bert_model_name="all-MiniLM-L6-v2",
        threshold=0.50,
    )
    metrics = model2.train(df)
    model2.save(MODEL2_SAVE_PATH)
    return {"model": "BERT Similarity", **metrics}


def compare_models(results: list):
    """Print a comparison table of both models."""
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'Model 1 (TFIDF+RF)':<22} {'Model 2 (BERT)':<20}")
    print("-"*62)
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]
    for metric in metrics:
        v1 = results[0].get(metric, 0)
        v2 = results[1].get(metric, 0) if len(results) > 1 else 0
        winner = "⬆ M1" if v1 > v2 else "⬆ M2"
        print(f"{metric:<20} {v1:<22.4f} {v2:<20.4f} {winner}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train Resume Screening Models")
    parser.add_argument("--data-only", action="store_true", help="Only generate data")
    parser.add_argument("--model1-only", action="store_true", help="Only train Model 1")
    parser.add_argument("--model2-only", action="store_true", help="Only train Model 2")
    parser.add_argument("--skip-generate", action="store_true", help="Use existing dataset")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="Number of training samples")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  RESUME SCREENING - ML TRAINING PIPELINE")
    print("🚀 "*15)
    #testing
    # Setup
    setup_mlflow()
    os.makedirs("saved_models", exist_ok=True)

    # Load / generate data
    df = load_or_generate_data(skip_generate=args.skip_generate)

    if args.data_only:
        print("\n[Done] Data generated. Exiting (--data-only mode).")
        return

    results = []

    # Train models
    if not args.model2_only:
        r1 = train_model1(df)
        results.append(r1)

    if not args.model1_only:
        r2 = train_model2(df)
        results.append(r2)

    # Compare
    if len(results) >= 2:
        compare_models(results)

    # Save results JSON (used by Jenkins / CI)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to {RESULTS_PATH}")

    print("\n✅ Training pipeline complete!")
    print(f"   View MLflow UI: mlflow ui --port 5000")
    print(f"   Run Streamlit:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
