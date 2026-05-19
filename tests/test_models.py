"""
Unit tests for SentimentModel and PredictionResult.

Tests cover:
  - PredictionResult field validation
  - MODEL_REGISTRY structure and completeness
  - SentimentModel error handling (invalid keys)
  - Mocked forward pass: tokenization → logits → softmax → label
  - Batch prediction output shape and types
  - Attention tensor presence and shape

Run with: pytest tests/test_models.py -v
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import fields

from src.models import (
    SentimentModel,
    PredictionResult,
    MODEL_REGISTRY,
)


# Helper
def make_mock_model_output(num_labels: int = 2, seq_len: int = 6, num_heads: int = 12):
    """
    Build a fake HuggingFace model output with logits and attentions,
    matching the shape that AutoModelForSequenceClassification returns.
    """
    logits = torch.zeros(1, num_labels)
    logits[0, 1] = 3.0 # push class 1 to be predicted 

    # attentions: tuple of (batch, heads, seq, seq), one per layer; we only use [-1]
    fake_attn = torch.rand(1, num_heads, seq_len, seq_len)
    fake_attn = fake_attn / fake_attn.sum(dim=-1, keepdim=True)  # normalize rows

    output = MagicMock()
    output.logits = logits
    output.attentions = (fake_attn,)  # tuple with one layer
    return output


def make_mock_tokenizer(seq_len: int = 6):
    """Return a mock tokenizer that produces fixed-size input tensors."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
    }
    tokenizer.convert_ids_to_tokens.return_value = (
        ["[CLS]", "this", "is", "great", "!", "[SEP]"][:seq_len]
    )
    return tokenizer


# PredictionResult 
class TestPredictionResult:

    def test_required_fields_exist(self):
        field_names = {f.name for f in fields(PredictionResult)}
        for required in ("model_name", "text", "label", "confidence", "all_scores", "latency_ms"):
            assert required in field_names, f"Missing field: {required}"

    def test_optional_fields_default(self):
        result = PredictionResult(
            model_name="distilbert",
            text="Hello",
            label="POSITIVE",
            confidence=0.9,
            all_scores={"POSITIVE": 0.9, "NEGATIVE": 0.1},
            latency_ms=20.0,
        )
        assert result.tokens == []
        assert result.attention is None

    def test_confidence_range(self):
        result = PredictionResult(
            model_name="distilbert",
            text="test",
            label="POSITIVE",
            confidence=0.95,
            all_scores={"POSITIVE": 0.95, "NEGATIVE": 0.05},
            latency_ms=10.0,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_all_scores_sum_to_one(self):
        result = PredictionResult(
            model_name="distilbert",
            text="test",
            label="POSITIVE",
            confidence=0.95,
            all_scores={"POSITIVE": 0.95, "NEGATIVE": 0.05},
            latency_ms=10.0,
        )
        total = sum(result.all_scores.values())
        assert abs(total - 1.0) < 1e-3

    def test_label_in_all_scores(self):
        result = PredictionResult(
            model_name="distilbert",
            text="test",
            label="POSITIVE",
            confidence=0.95,
            all_scores={"POSITIVE": 0.95, "NEGATIVE": 0.05},
            latency_ms=10.0,
        )
        assert result.label in result.all_scores

    def test_latency_is_positive(self):
        result = PredictionResult(
            model_name="distilbert",
            text="test",
            label="POSITIVE",
            confidence=0.9,
            all_scores={"POSITIVE": 0.9, "NEGATIVE": 0.1},
            latency_ms=12.5,
        )
        assert result.latency_ms > 0


# MODEL_REGISTRY 
class TestModelRegistry:

    def test_all_expected_keys_present(self):
        for key in ("distilbert", "roberta-twitter", "finbert"):
            assert key in MODEL_REGISTRY

    def test_each_entry_has_required_keys(self):
        for model_key, config in MODEL_REGISTRY.items():
            assert "model_id" in config,    f"{model_key}: missing 'model_id'"
            assert "labels" in config,      f"{model_key}: missing 'labels'"
            assert "description" in config, f"{model_key}: missing 'description'"

    def test_labels_are_dicts_with_int_keys(self):
        for model_key, config in MODEL_REGISTRY.items():
            labels = config["labels"]
            assert isinstance(labels, dict), f"{model_key}: 'labels' must be a dict"
            for k in labels:
                assert isinstance(k, int), f"{model_key}: label keys must be ints, got {type(k)}"

    def test_model_ids_are_nonempty_strings(self):
        for model_key, config in MODEL_REGISTRY.items():
            assert isinstance(config["model_id"], str) and len(config["model_id"]) > 0

    def test_label_indices_are_contiguous(self):
        """Label indices should start at 0 and be contiguous (0, 1, 2, ...)."""
        for model_key, config in MODEL_REGISTRY.items():
            indices = sorted(config["labels"].keys())
            assert indices == list(range(len(indices))), (
                f"{model_key}: label indices {indices} are not contiguous from 0"
            )


# SentimentModel (mocked)
class TestSentimentModel:

    def test_invalid_key_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model key"):
            SentimentModel("totally-fake-model")

    def test_error_message_lists_valid_keys(self):
        with pytest.raises(ValueError) as exc_info:
            SentimentModel("bad-key")
        error_msg = str(exc_info.value)
        for valid_key in MODEL_REGISTRY:
            assert valid_key in error_msg

    @patch("src.models.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_predict_returns_prediction_result(self, mock_tok_cls, mock_model_cls):
        seq_len = 6
        mock_tok_cls.return_value = make_mock_tokenizer(seq_len)
        mock_model_cls.return_value = MagicMock(
            return_value=make_mock_model_output(num_labels=2, seq_len=seq_len),
            eval=MagicMock(return_value=None),
        )
        mock_model_cls.return_value.to.return_value = mock_model_cls.return_value

        model = SentimentModel("distilbert", device="cpu")
        result = model.predict("This is great!")

        assert isinstance(result, PredictionResult)
        assert result.model_name == "distilbert"
        assert result.label in MODEL_REGISTRY["distilbert"]["labels"].values()
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0

    @patch("src.models.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_predict_confidence_matches_label(self, mock_tok_cls, mock_model_cls):
        """The confidence score should equal the probability of the predicted label."""
        seq_len = 6
        mock_tok_cls.return_value = make_mock_tokenizer(seq_len)
        mock_model_cls.return_value = MagicMock(
            return_value=make_mock_model_output(num_labels=2, seq_len=seq_len),
            eval=MagicMock(return_value=None),
        )
        mock_model_cls.return_value.to.return_value = mock_model_cls.return_value

        model = SentimentModel("distilbert", device="cpu")
        result = model.predict("Testing confidence alignment.")

        assert abs(result.confidence - result.all_scores[result.label]) < 1e-4

    @patch("src.models.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_predict_all_scores_keys_match_registry(self, mock_tok_cls, mock_model_cls):
        seq_len = 6
        mock_tok_cls.return_value = make_mock_tokenizer(seq_len)
        mock_model_cls.return_value = MagicMock(
            return_value=make_mock_model_output(num_labels=2, seq_len=seq_len),
            eval=MagicMock(return_value=None),
        )
        mock_model_cls.return_value.to.return_value = mock_model_cls.return_value

        model = SentimentModel("distilbert", device="cpu")
        result = model.predict("Test text.")

        expected_labels = set(MODEL_REGISTRY["distilbert"]["labels"].values())
        assert set(result.all_scores.keys()) == expected_labels

    @patch("src.models.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_predict_returns_attention_tensor(self, mock_tok_cls, mock_model_cls):
        seq_len = 6
        mock_tok_cls.return_value = make_mock_tokenizer(seq_len)
        mock_model_cls.return_value = MagicMock(
            return_value=make_mock_model_output(num_labels=2, seq_len=seq_len),
            eval=MagicMock(return_value=None),
        )
        mock_model_cls.return_value.to.return_value = mock_model_cls.return_value

        model = SentimentModel("distilbert", device="cpu")
        result = model.predict("Attention test.")

        assert result.attention is not None
        assert isinstance(result.attention, torch.Tensor)
        # Should be (seq_len, seq_len) averaged over heads
        assert result.attention.shape == (seq_len, seq_len)

    @patch("src.models.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_predict_batch_returns_correct_count(self, mock_tok_cls, mock_model_cls):
        seq_len = 6
        mock_tok_cls.return_value = make_mock_tokenizer(seq_len)
        mock_model_cls.return_value = MagicMock(
            return_value=make_mock_model_output(num_labels=2, seq_len=seq_len),
            eval=MagicMock(return_value=None),
        )
        mock_model_cls.return_value.to.return_value = mock_model_cls.return_value

        model = SentimentModel("distilbert", device="cpu")
        texts = ["Text one.", "Text two.", "Text three.", "Text four.", "Text five."]
        results = model.predict_batch(texts, batch_size=2)

        assert len(results) == len(texts)
        for r in results:
            assert isinstance(r, PredictionResult)

    @patch("src.models.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_predict_batch_all_results_have_correct_model_name(self, mock_tok_cls, mock_model_cls):
        seq_len = 6
        mock_tok_cls.return_value = make_mock_tokenizer(seq_len)
        mock_model_cls.return_value = MagicMock(
            return_value=make_mock_model_output(num_labels=2, seq_len=seq_len),
            eval=MagicMock(return_value=None),
        )
        mock_model_cls.return_value.to.return_value = mock_model_cls.return_value

        model = SentimentModel("distilbert", device="cpu")
        results = model.predict_batch(["Hello.", "World."])

        for r in results:
            assert r.model_name == "distilbert"

    @patch("src.models.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_finbert_three_label_output(self, mock_tok_cls, mock_model_cls):
        """FinBERT has 3 labels — verify all_scores has 3 entries."""
        seq_len = 5
        mock_tok_cls.return_value = make_mock_tokenizer(seq_len)
        mock_model_cls.return_value = MagicMock(
            return_value=make_mock_model_output(num_labels=3, seq_len=seq_len),
            eval=MagicMock(return_value=None),
        )
        mock_model_cls.return_value.to.return_value = mock_model_cls.return_value

        model = SentimentModel("finbert", device="cpu")
        result = model.predict("Earnings per share rose 4% year over year.")

        assert len(result.all_scores) == 3
        expected = set(MODEL_REGISTRY["finbert"]["labels"].values())
        assert set(result.all_scores.keys()) == expected
