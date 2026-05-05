"""
model2_bert.py
---------------
Model 2: BERT-based Semantic Similarity Resume Screener
- Uses sentence-transformers (all-MiniLM-L6-v2) to encode resume & JD
- Extracts semantic + keyword features
- Trains a calibrated Logistic Regression on top of BERT embeddings
- Returns a 0-100 match score and Shortlist/Reject verdict
- Logs all params & metrics to MLflow
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline as SKPipeline
import warnings
warnings.filterwarnings("ignore")


class BERTSimilarityScreener:
    """
    Resume Screener using BERT sentence embeddings.

    Workflow:
        1. Encode resume_text and job_desc using sentence-transformers
        2. Compute cosine similarity between embeddings
        3. Extract keyword overlap features
        4. Train Logistic Regression on [cosine_sim, overlap_features]
        5. Convert probability → 0-100 score
    """

    def __init__(
        self,
        bert_model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.50,
        experiment_name: str = "Resume_Screening_Model2_BERT",
    ):
        self.bert_model_name = bert_model_name
        self.threshold = threshold
        self.experiment_name = experiment_name
        self.model_name = "BERT Similarity + Logistic Regression"

        self._encoder = None           # sentence-transformers model
        self.classifier = SKPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                random_state=42,
            )),
        ])
        self.is_trained = False
        self.run_id = None

    # ─── Lazy-load BERT encoder ───────────────────────────────────────────────

    @property
    def encoder(self):
        """Lazy-load sentence-transformers model (downloads on first use)."""
        if self._encoder is None:
            print(f"[Model 2] Loading BERT model: {self.bert_model_name} ...")
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.bert_model_name)
            print(f"[Model 2] BERT model loaded ✅")
        return self._encoder

    # ─── Feature Extraction ───────────────────────────────────────────────────

    def _encode_texts(self, texts: list) -> np.ndarray:
        """Encode a list of texts using BERT."""
        return self.encoder.encode(texts, show_progress_bar=False, batch_size=32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _keyword_overlap(resume_text: str, job_desc: str) -> float:
        """Compute word-level overlap ratio between resume and JD."""
        resume_words = set(resume_text.lower().split())
        jd_words = set(job_desc.lower().split())
        # Remove stopwords (simple list)
        stopwords = {
            "the", "a", "an", "and", "or", "for", "in", "on", "at", "to",
            "of", "with", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "will", "would", "can", "could",
        }
        resume_words -= stopwords
        jd_words -= stopwords
        if not jd_words:
            return 0.0
        overlap = resume_words & jd_words
        return len(overlap) / len(jd_words)

    @staticmethod
    def _length_ratio(resume_text: str, job_desc: str) -> float:
        """Ratio of resume length to JD length (capped at 1)."""
        r_len = len(resume_text.split())
        j_len = max(len(job_desc.split()), 1)
        return min(r_len / j_len, 3.0) / 3.0  # normalized to [0, 1]

    def _extract_features(
        self,
        resume_texts: list,
        job_descs: list,
    ) -> np.ndarray:
        """
        Extract feature matrix for classifier.
        Features:
            [0] cosine_similarity  — BERT semantic similarity
            [1] keyword_overlap    — word overlap ratio
            [2] length_ratio       — resume/JD length ratio
            [3] combined_sim       — cosine_sim * keyword_overlap
        """
        print(f"[Model 2] Encoding {len(resume_texts)} resumes with BERT...")
        resume_embs = self._encode_texts(resume_texts)
        jd_embs = self._encode_texts(job_descs)

        features = []
        for r_emb, j_emb, r_text, j_text in zip(resume_embs, jd_embs, resume_texts, job_descs):
            cos_sim = self._cosine_similarity(r_emb, j_emb)
            kw_overlap = self._keyword_overlap(r_text, j_text)
            len_ratio = self._length_ratio(r_text, j_text)
            combined = cos_sim * kw_overlap

            features.append([cos_sim, kw_overlap, len_ratio, combined])

        return np.array(features, dtype=np.float32)

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Train the BERT-based screener and log to MLflow.

        Args:
            df: DataFrame with columns [resume_text, job_desc, label]
            test_size: fraction held out for evaluation

        Returns:
            dict of evaluation metrics
        """
        X_features = self._extract_features(
            df["resume_text"].tolist(),
            df["job_desc"].tolist(),
        )
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=y
        )

        # ── MLflow tracking ──────────────────────────────────────────────────
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="BERT_Similarity_Training") as run:
            self.run_id = run.info.run_id

            # Log hyperparameters
            mlflow.log_params({
                "model_type": "BERT + Logistic Regression",
                "bert_model": self.bert_model_name,
                "threshold": self.threshold,
                "features": "cosine_sim, keyword_overlap, length_ratio, combined_sim",
                "classifier": "LogisticRegression(class_weight=balanced, C=1.0)",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "total_samples": len(df),
            })

            # Train classifier on BERT features
            print(f"\n[Model 2] Training Logistic Regression on BERT features ({len(X_train)} samples)...")
            self.classifier.fit(X_train, y_train)
            self.is_trained = True

            # Evaluate
            metrics = self._evaluate(X_test, y_test)

            # Cross-val
            cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5, scoring="f1")
            metrics["cv_f1_mean"] = float(cv_scores.mean())
            metrics["cv_f1_std"] = float(cv_scores.std())

            # Compute and log mean cosine similarity
            cos_sims = X_features[:, 0]
            mlflow.log_metrics({
                **metrics,
                "mean_cosine_similarity": float(cos_sims.mean()),
                "mean_keyword_overlap": float(X_features[:, 1].mean()),
            })

            # Log the classifier
            mlflow.sklearn.log_model(
                self.classifier,
                artifact_path="bert_classifier",
                registered_model_name="ResumeScreener_BERT",
            )

            print(f"[Model 2] ✅ Training complete!")
            print(f"[Model 2] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
            print(f"[Model 2] CV F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
            print(f"[Model 2] Mean Cosine Similarity: {cos_sims.mean():.4f}")
            print(f"[Model 2] MLflow Run ID: {self.run_id}")

        return metrics

    def _evaluate(self, X_test, y_test) -> dict:
        """Evaluate on test features."""
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)[:, 1]

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_proba)),
        }

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict(self, resume_text: str, job_desc: str) -> dict:
        """
        Predict match score for a single resume-JD pair.

        Returns:
            dict with score (0-100), probability, shortlisted, cosine_similarity
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        # Extract features for single pair
        features = self._extract_features([resume_text], [job_desc])
        proba = self.classifier.predict_proba(features)[0][1]
        score = round(proba * 100, 2)
        shortlisted = proba >= self.threshold
        cos_sim = float(features[0][0])

        return {
            "model_name": self.model_name,
            "score": score,
            "probability": round(float(proba), 4),
            "shortlisted": bool(shortlisted),
            "verdict": "✅ Shortlisted" if shortlisted else "❌ Rejected",
            "threshold_used": self.threshold,
            "cosine_similarity": round(cos_sim * 100, 2),
            "keyword_overlap": round(float(features[0][1]) * 100, 2),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict on a DataFrame with resume_text and job_desc columns."""
        results = [self.predict(r, j) for r, j in zip(df["resume_text"], df["job_desc"])]
        return pd.DataFrame(results)

    # ─── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str = "saved_models/model2_bert.pkl"):
        """Save the trained classifier (BERT encoder is not saved — it's loaded fresh)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "classifier": self.classifier,
            "bert_model_name": self.bert_model_name,
            "threshold": self.threshold,
            "run_id": self.run_id,
            "model_name": self.model_name,
        }, path)
        print(f"[Model 2] Saved to {path}")

    def load(self, path: str = "saved_models/model2_bert.pkl"):
        """Load a previously saved model."""
        data = joblib.load(path)
        self.classifier = data["classifier"]
        self.bert_model_name = data["bert_model_name"]
        self.threshold = data["threshold"]
        self.run_id = data.get("run_id")
        self.is_trained = True
        print(f"[Model 2] Loaded from {path}")
        return self
