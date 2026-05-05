"""
model1_tfidf_rf.py
-------------------
Model 1: TF-IDF + Random Forest Resume Screener
- Converts resume+JD text into TF-IDF features
- Trains a Random Forest classifier to predict match/no-match
- Returns a 0-100 match score and Shortlist/Reject verdict
- Logs all params & metrics to MLflow
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


class TFIDFRandomForestScreener:
    """
    Resume Screener using TF-IDF features + Random Forest classifier.

    Workflow:
        1. Concatenate resume_text + [SEP] + job_desc into a single string
        2. TF-IDF vectorize (captures keyword overlap)
        3. Random Forest predicts match probability
        4. Convert probability → 0-100 score
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        max_features_tfidf: int = 8000,
        threshold: float = 0.50,
        experiment_name: str = "Resume_Screening_Model1_TFIDF_RF",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features_tfidf = max_features_tfidf
        self.threshold = threshold
        self.experiment_name = experiment_name
        self.model_name = "TF-IDF + Random Forest"

        # sklearn pipeline: vectorizer → classifier
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features_tfidf,
                ngram_range=(1, 2),          # unigrams + bigrams
                stop_words="english",
                sublinear_tf=True,           # log normalization
            )),
            ("clf", RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self.is_trained = False
        self.run_id = None

    # ─── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _combine_text(resume_text: str, job_desc: str) -> str:
        """Combine resume and job description into a single feature string."""
        return f"{resume_text.strip()} [SEP] {job_desc.strip()}"

    def _prepare_X(self, df: pd.DataFrame) -> list:
        return [self._combine_text(r, j) for r, j in zip(df["resume_text"], df["job_desc"])]

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Train the model and log everything to MLflow.

        Args:
            df: DataFrame with columns [resume_text, job_desc, label]
            test_size: fraction held out for evaluation

        Returns:
            dict of evaluation metrics
        """
        X_text = self._prepare_X(df)
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=42, stratify=y
        )

        # ── MLflow tracking ──────────────────────────────────────────────────
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="TFIDF_RF_Training") as run:
            self.run_id = run.info.run_id

            # Log hyperparameters
            mlflow.log_params({
                "model_type": "TF-IDF + Random Forest",
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "max_features_tfidf": self.max_features_tfidf,
                "ngram_range": "(1,2)",
                "threshold": self.threshold,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "total_samples": len(df),
            })

            # Train
            print(f"\n[Model 1] Training TF-IDF + Random Forest on {len(X_train)} samples...")
            self.pipeline.fit(X_train, y_train)
            self.is_trained = True

            # Evaluate
            metrics = self._evaluate(X_test, y_test)

            # Cross-validation score
            cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring="f1")
            metrics["cv_f1_mean"] = float(cv_scores.mean())
            metrics["cv_f1_std"] = float(cv_scores.std())

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log the model artifact
            mlflow.sklearn.log_model(
                self.pipeline,
                artifact_path="tfidf_rf_model",
                registered_model_name="ResumeScreener_TFIDF_RF",
            )

            # Log feature importances as artifact
            self._log_feature_importance(run.info.artifact_uri)

            print(f"[Model 1] ✅ Training complete!")
            print(f"[Model 1] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
            print(f"[Model 1] CV F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
            print(f"[Model 1] MLflow Run ID: {self.run_id}")

        return metrics

    def _evaluate(self, X_test, y_test) -> dict:
        """Evaluate model on test data."""
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_proba)),
        }

    def _log_feature_importance(self, artifact_uri: str):
        """Log top feature importances to MLflow."""
        try:
            vectorizer = self.pipeline.named_steps["tfidf"]
            rf = self.pipeline.named_steps["clf"]
            feature_names = vectorizer.get_feature_names_out()
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[-20:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in top_indices]

            lines = ["feature,importance\n"]
            for feat, imp in top_features:
                lines.append(f"{feat},{imp:.6f}\n")

            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.writelines(lines)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, "feature_importance")
            os.remove(tmp_path)
        except Exception:
            pass  # Non-critical

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict(self, resume_text: str, job_desc: str) -> dict:
        """
        Predict match score for a single resume-JD pair.

        Returns:
            dict with keys: score (0-100), probability, shortlisted (bool), model_name
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        combined = self._combine_text(resume_text, job_desc)
        proba = self.pipeline.predict_proba([combined])[0][1]
        score = round(proba * 100, 2)
        shortlisted = proba >= self.threshold

        return {
            "model_name": self.model_name,
            "score": score,
            "probability": round(float(proba), 4),
            "shortlisted": bool(shortlisted),
            "verdict": "✅ Shortlisted" if shortlisted else "❌ Rejected",
            "threshold_used": self.threshold,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict on a DataFrame with resume_text and job_desc columns."""
        results = [self.predict(r, j) for r, j in zip(df["resume_text"], df["job_desc"])]
        return pd.DataFrame(results)

    # ─── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str = "saved_models/model1_tfidf_rf.pkl"):
        """Save the trained model pipeline."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "threshold": self.threshold,
            "run_id": self.run_id,
            "model_name": self.model_name,
        }, path)
        print(f"[Model 1] Saved to {path}")

    def load(self, path: str = "saved_models/model1_tfidf_rf.pkl"):
        """Load a previously saved model."""
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.threshold = data["threshold"]
        self.run_id = data.get("run_id")
        self.is_trained = True
        print(f"[Model 1] Loaded from {path}")
        return self
