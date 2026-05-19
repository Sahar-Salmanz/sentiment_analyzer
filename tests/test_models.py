"""
Unit tests for SentimentModel and PredictionResult.

Tests cover:
  - PredictionResult field validation
  - MODEL_REGISTRY structure and completeness
  - SentimentModel error handling (invalid keys)
  - Mocked forward pass: tokenization → logits → softmax → label
  - Batch prediction output shape and types
  - Attention tensor presence and shape

Run with: pytest tests/ -v
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
    """_summary_

    :param num_labels: _description_, defaults to 2
    :type num_labels: int, optional
    :param seq_len: _description_, defaults to 6
    :type seq_len: int, optional
    :param num_heads: _description_, defaults to 12
    :type num_heads: int, optional
    """
    logits = torch.zeros(1, num_labels)
    logits[0, 1] = 3.0 # push class 1 to be predicted 

    fake_attn = torch.rand(1, num_heads, seq_len, seq_len)