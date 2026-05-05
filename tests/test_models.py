"""
test_models.py
---------------
Unit tests for both ML models.
Jenkins runs these tests automatically after training.

Run:
    pytest tests/ -v
    pytest tests/ -v --cov=models
"""

import os
import sys
import pytest
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_RESUME_MATCH = """
John Doe | Senior Data Scientist
Python, Machine Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy,
SQL, NLP, Docker, AWS, MLflow, Feature Engineering, A/B Testing.
5 years experience at Google building recommendation systems.
M.Sc Data Science from IIT Bombay, 2020.
"""

SAMPLE_RESUME_NOMATCH = """
Jane Smith | Marketing Manager
Digital Marketing, SEO, SEM, Content Strategy, Social Media, HubSpot, CRM,
Google Analytics, Facebook Ads, Brand Management, Copywriting.
4 years in digital marketing at Dentsu. MBA from XLRI.
"""

SAMPLE_JD_DS = """
Job Title: Data Scientist
Required: Python, Machine Learning, TensorFlow, SQL, Statistics, NLP, Docker.
We need 3+ years of experience in ML model building and deployment.
"""


@pytest.fixture
def sample_df():
    """Create a small training DataFrame."""
    records = []
    # Match pairs
    for _ in range(30):
        records.append({
            "resume_text": SAMPLE_RESUME_MATCH,
            "job_desc": SAMPLE_JD_DS,
            "label": 1,
        })
    # No-match pairs
    for _ in range(30):
        records.append({
            "resume_text": SAMPLE_RESUME_NOMATCH,
            "job_desc": SAMPLE_JD_DS,
            "label": 0,
        })
    return pd.DataFrame(records)


# ─── Model 1 Tests ────────────────────────────────────────────────────────────

class TestTFIDFRandomForestScreener:

    def test_import(self):
        """Test Model 1 can be imported."""
        from models.model1_tfidf_rf import TFIDFRandomForestScreener
        assert TFIDFRandomForestScreener is not None

    def test_instantiation(self):
        """Test Model 1 instantiation with default params."""
        from models.model1_tfidf_rf import TFIDFRandomForestScreener
        model = TFIDFRandomForestScreener()
        assert model.n_estimators == 200
        assert model.threshold == 0.50
        assert not model.is_trained

    def test_training(self, sample_df):
        """Test Model 1 training returns metrics."""
        from models.model1_tfidf_rf import TFIDFRandomForestScreener
        model = TFIDFRandomForestScreener(n_estimators=10, max_depth=5)  # small for speed
        metrics = model.train(sample_df)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0
        assert model.is_trained

    def test_prediction_output_format(self, sample_df):
        """Test prediction returns correct dict format."""
        from models.model1_tfidf_rf import TFIDFRandomForestScreener
        model = TFIDFRandomForestScreener(n_estimators=10, max_depth=5)
        model.train(sample_df)

        result = model.predict(SAMPLE_RESUME_MATCH, SAMPLE_JD_DS)
        assert "score" in result
        assert "probability" in result
        assert "shortlisted" in result
        assert "verdict" in result
        assert 0 <= result["score"] <= 100
        assert 0.0 <= result["probability"] <= 1.0
        assert isinstance(result["shortlisted"], bool)

    def test_match_resume_scores_higher(self, sample_df):
        """Matching resume should score higher than non-matching."""
        from models.model1_tfidf_rf import TFIDFRandomForestScreener
        model = TFIDFRandomForestScreener(n_estimators=50, max_depth=10)
        model.train(sample_df)

        result_match = model.predict(SAMPLE_RESUME_MATCH, SAMPLE_JD_DS)
        result_nomatch = model.predict(SAMPLE_RESUME_NOMATCH, SAMPLE_JD_DS)
        assert result_match["score"] > result_nomatch["score"], (
            f"Match score ({result_match['score']}) should be > no-match ({result_nomatch['score']})"
        )

    def test_save_and_load(self, sample_df, tmp_path):
        """Test model save/load cycle."""
        from models.model1_tfidf_rf import TFIDFRandomForestScreener
        model = TFIDFRandomForestScreener(n_estimators=10, max_depth=5)
        model.train(sample_df)

        save_path = str(tmp_path / "model1_test.pkl")
        model.save(save_path)
        assert os.path.exists(save_path)

        model2 = TFIDFRandomForestScreener()
        model2.load(save_path)
        assert model2.is_trained

        result = model2.predict(SAMPLE_RESUME_MATCH, SAMPLE_JD_DS)
        assert 0 <= result["score"] <= 100

    def test_predict_before_train_raises(self):
        """Predicting before training should raise RuntimeError."""
        from models.model1_tfidf_rf import TFIDFRandomForestScreener
        model = TFIDFRandomForestScreener()
        with pytest.raises(RuntimeError):
            model.predict("some resume", "some jd")


# ─── Model 2 Tests ────────────────────────────────────────────────────────────

class TestBERTSimilarityScreener:

    def test_import(self):
        """Test Model 2 can be imported."""
        from models.model2_bert import BERTSimilarityScreener
        assert BERTSimilarityScreener is not None

    def test_instantiation(self):
        """Test Model 2 instantiation with default params."""
        from models.model2_bert import BERTSimilarityScreener
        model = BERTSimilarityScreener()
        assert model.bert_model_name == "all-MiniLM-L6-v2"
        assert model.threshold == 0.50
        assert not model.is_trained

    def test_training(self, sample_df):
        """Test Model 2 training returns metrics."""
        from models.model2_bert import BERTSimilarityScreener
        model = BERTSimilarityScreener()
        metrics = model.train(sample_df)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert model.is_trained

    def test_prediction_output_format(self, sample_df):
        """Test prediction returns correct dict with BERT-specific fields."""
        from models.model2_bert import BERTSimilarityScreener
        model = BERTSimilarityScreener()
        model.train(sample_df)

        result = model.predict(SAMPLE_RESUME_MATCH, SAMPLE_JD_DS)
        assert "score" in result
        assert "probability" in result
        assert "shortlisted" in result
        assert "verdict" in result
        assert "cosine_similarity" in result
        assert "keyword_overlap" in result
        assert 0 <= result["score"] <= 100
        assert 0 <= result["cosine_similarity"] <= 100

    def test_cosine_similarity_static(self):
        """Test cosine similarity static method."""
        from models.model2_bert import BERTSimilarityScreener
        import numpy as np
        model = BERTSimilarityScreener()

        # Identical vectors → similarity = 1
        a = np.array([1.0, 0.0, 0.0])
        assert abs(model._cosine_similarity(a, a) - 1.0) < 1e-6

        # Orthogonal vectors → similarity = 0
        b = np.array([0.0, 1.0, 0.0])
        assert abs(model._cosine_similarity(a, b)) < 1e-6

    def test_keyword_overlap_static(self):
        """Test keyword overlap calculation."""
        from models.model2_bert import BERTSimilarityScreener
        model = BERTSimilarityScreener()

        resume = "Python Machine Learning TensorFlow"
        jd = "Python Machine Learning Docker"
        overlap = model._keyword_overlap(resume, jd)
        assert 0.0 < overlap < 1.0  # "Python" and "Machine" and "Learning" overlap

        # No overlap
        resume2 = "Marketing SEO Copywriting"
        overlap2 = model._keyword_overlap(resume2, jd)
        assert overlap2 < overlap

    def test_predict_before_train_raises(self):
        """Predicting before training should raise RuntimeError."""
        from models.model2_bert import BERTSimilarityScreener
        model = BERTSimilarityScreener()
        with pytest.raises(RuntimeError):
            model.predict("some resume", "some jd")

    def test_save_and_load(self, sample_df, tmp_path):
        """Test model save/load cycle."""
        from models.model2_bert import BERTSimilarityScreener
        model = BERTSimilarityScreener()
        model.train(sample_df)

        save_path = str(tmp_path / "model2_test.pkl")
        model.save(save_path)
        assert os.path.exists(save_path)

        model2 = BERTSimilarityScreener()
        model2.load(save_path)
        assert model2.is_trained

        result = model2.predict(SAMPLE_RESUME_MATCH, SAMPLE_JD_DS)
        assert 0 <= result["score"] <= 100


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestIntegration:

    def test_training_results_json_exists(self):
        """Check training results file exists (written by train.py)."""
        if os.path.exists("training_results.json"):
            with open("training_results.json") as f:
                data = json.load(f)
            assert isinstance(data, list)
            for result in data:
                assert "accuracy" in result
                assert "f1" in result

    def test_saved_models_exist(self):
        """Verify saved model files exist after training."""
        model1_path = "saved_models/model1_tfidf_rf.pkl"
        model2_path = "saved_models/model2_bert.pkl"
        assert os.path.exists(model1_path), f"Model 1 not found at {model1_path}"
        assert os.path.exists(model2_path), f"Model 2 not found at {model2_path}"

    def test_dataset_exists(self):
        """Verify training dataset was generated."""
        data_path = "data/resume_dataset.csv"
        assert os.path.exists(data_path), f"Dataset not found at {data_path}"
        df = pd.read_csv(data_path)
        assert "resume_text" in df.columns
        assert "job_desc" in df.columns
        assert "label" in df.columns
        assert len(df) > 0
