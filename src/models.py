"""
Model registry and loader for multi-model sentiment analysis.

Supports loading any HuggingFace sequence classification model by name.
Wraps each model with a consistent interface for inference and attention extraction.
"""

import time
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Supported models with their lable mapping
MODEL_REGISTRY = {
    "distilbert": {
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "labels": {0: "NEGATIVE", 1: "POSITIVE"},
        "description": "DistilBERT fine-tuned on SST-2 (general sentiment)",
    }, 
    "roberta-twitter": {
        "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "labels": {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
        "description": "RoBERTa fine-tuned on Twitter data (social media sentiment)",
    },
    "finbert": {
        "model_id": "ProsusAI/finbert",
        "labels": {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"},
        "description": "FinBERT fine-tuned on financial news (domain-specific)",
    },

}


@dataclass
class PredictionResult:
    model_name: str
    text: str
    label: str
    confidence: float
    latency_ms: float
    all_scores: dict
    tokens: list = field(default_factory=list) # ensure every instance gets its own copy
    attention: Optional[torch.Tensor] = None

class SentimentModel:
    """
    Wraps a HuggingFace model for sentiment analysis with a consistent interface.
    Supports single and batch inference, plus attention weight extraction.
    """
    def __init__(self, model_key: str, device: Optional[str] = None):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}"
            )
        
        self.model_key = model_key
        config = MODEL_REGISTRY[model_key]
        self.model_id = config["model_id"]
        self.label_map = config["labels"]
        self.description = config["description"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_key} from '{self.model_id}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id) # initialize tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, 
            output_attentions=True,
        ).to(self.device)
        self.model.eval()
        print(f"  ✓ {model_key} ready.")

    def predict(self, text: str, max_length: int = 512) -> PredictionResult:
        """Run inference on a single text string."""
        start = time.perf_counter()

        # inputs = self.tokenizer(
        #     text, 
        #     return_tensors="pt",
        #     truncation=True,
        #     max_length= max_length,
        #     padding=True,
        # ).to(self.device)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = self.model(**inputs) # goes to the forwad call method defined in AutoModelForSequenceClassification
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0] # dim=-1 > apply softmax along the last dimension
        predicted_idx = probs.argmax().item()

        # Average attention across all heads in the last layer
        last_layer_attn = outputs.attentions[-1]
        avg_attn = last_layer_attn[0].mean(dim=0)

        latency_ms = (time.perf_counter() - start) * 1000

        all_scores = {
            self.label_map[i]: round(probs[i].item(), 4) for i in range(len(self.label_map))
        }

        return PredictionResult(
            model_name = self.model_key,
            text = text,
            label = self.label_map[predicted_idx], 
            confidence = round(probs[predicted_idx].item(), 4),
            latency_ms = round(latency_ms, 2), 
            all_scores = all_scores, 
            tokens = tokens, 
            attention = avg_attn.cpu(),
        )
    
    def predict_batch(self, texts: str, batch_size: int = 16) -> list[PredictionResult]:
        """Run batched inference for efficiency on large datasets."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.predict(text))
        return results