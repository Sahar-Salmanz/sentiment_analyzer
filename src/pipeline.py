"""
Multi-model inference pipeline.

Loads multiple models at once and provides a unified interface to run
predictions across all of them, compare results, and aggregate outputs.
"""

import pandas as pd
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import classification_report

from .models import SentimentModel, PredictionResult, MODEL_REGISTRY


class SentimentPipeline:
    def __init__(self, model_keys: Optional[list[str]] = None, device: Optional[str] = None):
        """
        :param model_keys: List of model keys to load. Defaults to all registered models.
        :param device: Torch device string ('cpu', 'cuda', 'mps').
        """
        if model_keys is None:
            model_keys = list(MODEL_REGISTRY.keys())

        self.models = {}
        for key in model_keys:
            self.models[key] = SentimentModel(key, device=device)

    def predict(self, text: str) -> dict[str, PredictionResult]:
        """
        Run all loaded models on a single text input.

        :return: Dict mapping model_key -> PredictionResult
        """
        return {key: model.predict(text) for key, model in self.models.items()}
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """ 
        Run inference on every row of a DataFrame column.
        Adds columns: {model_key}_label, {model_key}_confidence, {model_key}_latency_ms
        for each loaded model.

        :param df: Input DataFrame.
        :param text_column: Name of the column containing input text.
        :return: The original DataFrame with new prediction columns appended.
        """
        results_by_model = {key: [] for key in self.models}

        texts = df[text_column].tolist()
        for text in tqdm(texts, desc="Running inference"):
            preds = self.predict(text)
            for key, result in preds.items():
                results_by_model[key].append(result)

        for key, results in results_by_model.items():
            df[f"{key}_label"] = [r.label for r in results]
            df[f"{key}_confidence"] = [r.confidence for r in results]
            df[f"{key}_latency_ms"] = [r.latency_ms for r in results]

        return df
    
    def benchmark(self, texts: list[str], true_labels: list[str], model_key: str) -> str:
        """
        Evaluate a single model against ground-truth labels.

        :return: A classification report string (precision, recall, F1 per class).
        """
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' is not loaded.")
        
        model = self.models[model_key]
        results = model.predict_batch(texts)
        predicted = [r.label for r in results]

        report = classification_report(true_labels, predicted)

        return report
    
    def compare(self, text: str) -> pd.DataFrame:
        """
        Pretty-print a comparison table of all models on a single input.

        :return: DataFrame with columns: model, label, confidence, latency_ms
        """
        preds = self.predict(text)
        rows = []
        for key, result in preds.items():
            rows.append({
                "model": key,
                "label": result.label, 
                "confidence": f"{result.confidence * 100:.1f}%",
                "latency_ms": f"{result.latency_ms:.1f}ms",
                "all_scores": result.all_scores,
            })
        return pd.DataFrame(rows)
