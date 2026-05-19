"""
Unit tests for the inference pipeline.

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# Fixtures
@pytest.fixture
def mock_prediction():
    from src.models import PredictionResult
    return PredictionResult(
        model_name="distilbert",
        text="This is great!",
        label="POSITIVE",
        confidence=0.98,
        all_scores={"NEGATIVE": 0.02, "POSITIVE": 0.98},
        latency_ms=45.3,
        tokens=["[CLS]", "this", "is", "great", "!", "[SEP]"],
        attention=None,
    )


# Unit tests: PredictionResult
def test_prediction_result_fields(mock_prediction):
    assert mock_prediction.label == "POSITIVE"
    assert mock_prediction.confidence == 0.98
    assert mock_prediction.model_name == "distilbert"
    assert "POSITIVE" in mock_prediction.all_scores
    assert "NEGATIVE" in mock_prediction.all_scores


def test_prediction_confidence_sum(mock_prediction):
    total = sum(mock_prediction.all_scores.values())
    assert abs(total - 1.0) < 1e-4, f"Probabilities should sum to ~1.0, got {total}"


# Unit tests: utils
def test_clean_text_strips_whitespace():
    from src.utils import clean_text
    assert clean_text("  hello world  ") == "hello world"


def test_clean_text_removes_urls():
    from src.utils import clean_text
    result = clean_text("Visit https://example.com for details")
    assert "https" not in result
    assert "example.com" not in result


def test_clean_text_removes_html():
    from src.utils import clean_text
    result = clean_text("<b>Bold text</b> and <i>italic</i>")
    assert "<b>" not in result
    assert "<i>" not in result
    assert "Bold text" in result


def test_clean_text_collapses_spaces():
    from src.utils import clean_text
    result = clean_text("too   many     spaces")
    assert "  " not in result


def test_results_to_dataframe(mock_prediction):
    from src.utils import results_to_dataframe
    df = results_to_dataframe([mock_prediction])
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns
    assert "confidence" in df.columns
    assert df.iloc[0]["label"] == "POSITIVE"


# Unit tests: MODEL_REGISTRY
def test_model_registry_keys():
    from src.models import MODEL_REGISTRY
    assert "distilbert" in MODEL_REGISTRY
    assert "roberta-twitter" in MODEL_REGISTRY
    assert "finbert" in MODEL_REGISTRY


def test_model_registry_structure():
    from src.models import MODEL_REGISTRY
    for key, config in MODEL_REGISTRY.items():
        assert "model_id" in config, f"Missing 'model_id' in {key}"
        assert "labels" in config, f"Missing 'labels' in {key}"
        assert isinstance(config["labels"], dict), f"'labels' should be a dict in {key}"


def test_invalid_model_key_raises():
    from src.models import SentimentModel
    with pytest.raises(ValueError, match="Unknown model key"):
        SentimentModel("nonexistent-model")


# Integration-style test (mocked, no actual model download)
def test_pipeline_compare_returns_dataframe():
    """
    Mocked integration test — verifies SentimentPipeline.compare() returns
    a DataFrame without actually loading transformer models.
    """
    from src.models import PredictionResult

    mock_result = PredictionResult(
        model_name="distilbert",
        text="Great product!",
        label="POSITIVE",
        confidence=0.95,
        all_scores={"NEGATIVE": 0.05, "POSITIVE": 0.95},
        latency_ms=30.0,
    )

    with patch("src.models.SentimentModel") as MockModel:
        instance = MockModel.return_value
        instance.predict.return_value = mock_result

        from src.pipeline import SentimentPipeline
        pipeline = SentimentPipeline.__new__(SentimentPipeline)
        pipeline.models = {"distilbert": instance}

        result_df = pipeline.compare("Great product!")
        assert isinstance(result_df, pd.DataFrame)
        assert "label" in result_df.columns
        assert result_df.iloc[0]["label"] == "POSITIVE"
