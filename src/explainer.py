"""
Model explainability for sentiment predictions.

Provides two explanation methods:
  1. Attention heatmap  — visualize which tokens the model attended to
  2. SHAP values        — token-level feature importance via KernelSHAP

Usage:
    from sentiment.explainer import AttentionExplainer, ShapExplainer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
from typing import Optional

from .models import SentimentModel, PredictionResult


class AttentionExplainer:
    """
    Renders a token-level attention heatmap from the last transformer layer.
    Shows the CLS token's attention over the input sequence — a rough proxy
    for which tokens influenced the final classification.
    """
    
    def __init__(self, model: SentimentModel):
        self.model = model 

    def explain(self, text: str, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate and return an attention heatmap for the given text.

        :param text: Input sentence.
        :param save_path: If provided, saves the figure to this path.
        :return: matplotlib Figure object.
        """
        result: PredictionResult = self.model.predict(text)

        if result.attention is None:
            raise RuntimeError("No attention tensors returned, Ensure output_attentions=True")
        
        # CLS token's attention over all other tokens
        cls_attn = result.attention[0].numpy()
        tokens = result.tokens

        # Strip [CLS] and [SEP] for cleaner display
        tokens_clean = [t for t in tokens if t not in ("[CLS]", "<s>", "</s>")]
        attn_clean = cls_attn[1 : len(tokens_clean) + 1]
        attn_clean = attn_clean / attn_clean.sum() # Normalize

        fig, ax = plt.subplots(figsize=(max(8, len(tokens_clean) + 0.6), 2.5))
        attn_matrix = attn_clean.reshape(1, -1)

        sns.heatmap(
            attn_matrix, 
            ax=ax,
            xticklabels=tokens_clean,
            yticklabels=["attention"], 
            cmap="Blues", 
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar=False,
        )

        ax.set_title(
            f'Attention heatmap — "{text[:60]}..."\n'
            f'Prediction: {result.label} ({result.confidence * 100:.1f}%)',
            fontsize=11,
            pad=12,
        )
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
    

class ShapExplainer:
    """
    Uses SHAP KernelExplainer to compute token-level feature importance.
    Slower than attention but more theoretically grounded.

    """
    def __init__(self, model: SentimentModel):
        self.model = model
        self._shap_explainer = None

    def _predict_probs(self, texts: list) -> np.ndarray:
        """Adapter for SHAP: takes a list of texts, returns probability arrays."""
        import torch

        all_probs = []
        for text in texts:
            inputs = self.model.tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            all_probs.append(probs)

        return np.array(all_probs)
    
    def explain(self, text: str, n_samples: int = 100) -> dict:
        """
        Compute SHAP values for each token in the input text.

        :param text: Input sentence.
        :param n_samples: Number of SHAP perturbation samples (higher = more accurate).
        :return: Dict with keys: tokens, shap_values, base_values, prediction
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Run `pip install shap` to use ShapExplainer.")

        result = self.model.predict(text)
        tokens = [t for t in result.tokens if t not in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]")]

        # Use tokens as "features" for SHAP
        masker = shap.maskers.Text(self.model.tokenizer)
        explainer = shap.Explainer(self._predict_proba, masker)
        shap_values = explainer([text], max_evals=n_samples)

        return {
            "tokens": tokens,
            "shap_values": shap_values,
            "prediction": result,
        }
    
    def plot(self, text: str, class_index: int = 1, save_path: Optional[str] = None) -> plt.Figure:
        """
        Render a bar chart of per-token SHAP values for a given class.

        :param text: Input text.
        :param class_index: Index of the target class (1 = POSITIVE for binary models).
        :param save_path: Optional file path to save the figure.
        """
        result = self.explain(text)
        shap_values = result["shap_values"]

        # shap plot returns its own figure
        fig = plt.figure()
        shap.plots.text(shap_values[:, :, class_index], display=False)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig