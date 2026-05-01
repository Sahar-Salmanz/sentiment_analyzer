# 🧠 NLP Sentiment Analysis Suite

[![CI](https://github.com/your-username/nlp-sentiment-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/nlp-sentiment-suite/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-structured NLP project for **multi-model sentiment analysis** with attention-based explainability. Compare three transformer models side-by-side, run batch inference on CSV datasets, and visualize what each model pays attention to — all through a clean Gradio web interface.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-model inference** | DistilBERT (SST-2), RoBERTa-Twitter, FinBERT run in parallel |
| **Attention heatmaps** | Visualize per-token attention weights from the last transformer layer |
| **SHAP explainability** | KernelSHAP for theoretically-grounded token importance scores |
| **Batch inference** | Upload a CSV and annotate every row with predictions + confidence |
| **Gradio web UI** | Interactive demo with model comparison table and attention plots |
| **Benchmarking** | Evaluate any model against ground-truth labels (precision/recall/F1) |
| **CI pipeline** | GitHub Actions runs pytest on every push |

---

## 🏗️ Project Structure

```
nlp-sentiment-suite/
├── src/sentiment/
│   ├── models.py        # Model loader, SentimentModel class, PredictionResult dataclass
│   ├── pipeline.py      # SentimentPipeline — multi-model orchestration
│   ├── explainer.py     # AttentionExplainer + ShapExplainer
│   └── utils.py         # Text cleaning, CSV loading, results formatting
├── app/
│   └── gradio_app.py    # Interactive Gradio web demo
├── tests/
│   └── test_pipeline.py # Unit + mocked integration tests
├── data/
│   └── sample_reviews.csv
└── .github/workflows/
    └── ci.yml
```

---

## 🚀 Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/nlp-sentiment-suite.git
cd nlp-sentiment-suite

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .               # Install as editable package
```

### 2. Run the Gradio web app

```bash
python app/gradio_app.py
```

Then open [http://localhost:7860](http://localhost:7860).

### 3. Use as a Python library

```python
from sentiment import SentimentPipeline, AttentionExplainer

# Load all three models
pipeline = SentimentPipeline()

# Compare predictions on a single text
df = pipeline.compare("Apple stock surged 12% after blowout earnings.")
print(df)

# Batch inference on a DataFrame
import pandas as pd
df = pd.read_csv("data/sample_reviews.csv")
results = pipeline.predict_dataframe(df, text_column="text")
print(results.head())

# Attention heatmap
explainer = AttentionExplainer(pipeline.models["distilbert"])
fig = explainer.explain("The product quality is outstanding.")
fig.savefig("attention_map.png", dpi=150, bbox_inches="tight")
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## 🤖 Models

| Key | Model ID | Labels | Best for |
|---|---|---|---|
| `distilbert` | `distilbert-base-uncased-finetuned-sst-2-english` | POS / NEG | General text |
| `roberta-twitter` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | POS / NEU / NEG | Social media |
| `finbert` | `ProsusAI/finbert` | POS / NEG / NEU | Financial news |

All models are loaded from the [Hugging Face Hub](https://huggingface.co/models) and cached locally on first run.

---

## 🔍 Explainability

### Attention heatmap

Extracts the CLS token's attention weights from the final transformer layer. A lightweight, fast proxy for token importance.

```python
from sentiment import SentimentModel, AttentionExplainer

model = SentimentModel("distilbert")
explainer = AttentionExplainer(model)
fig = explainer.explain("The results were deeply disappointing.")
fig.show()
```

### SHAP values

Uses KernelSHAP to compute theoretically-grounded token importance through perturbation. Slower but more rigorous.

```bash
pip install shap
```

```python
from sentiment import SentimentModel, ShapExplainer

model = SentimentModel("distilbert")
explainer = ShapExplainer(model)
result = explainer.explain("Outstanding performance this quarter.")
# result["shap_values"] contains SHAP attribution per token
```

---

## 📊 Benchmarking

Evaluate a model against labeled data and get a full classification report:

```python
from sentiment import SentimentPipeline

pipeline = SentimentPipeline(model_keys=["distilbert"])

texts = ["Great!", "Terrible.", "It was okay."]
labels = ["POSITIVE", "NEGATIVE", "NEGATIVE"]

report = pipeline.benchmark(texts, labels, model_key="distilbert")
print(report)
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

Tests use `unittest.mock` to patch model loading — no actual downloads required during CI.

---

## 🛠️ Tech Stack

- [PyTorch](https://pytorch.org/) — tensor operations and model inference
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — pretrained models and tokenizers
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — dataset loading
- [SHAP](https://shap.readthedocs.io/) — model explainability
- [Gradio](https://gradio.app/) — interactive web demo
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics
- [GitHub Actions](https://docs.github.com/en/actions) — CI/CD

---

## 📈 Roadmap

- [ ] Fine-tune DistilBERT on a custom domain dataset
- [ ] Add LIME explainability as an alternative to SHAP
- [ ] Docker container for one-command deployment
- [ ] Streamlit dashboard variant
- [ ] REST API wrapper using FastAPI

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
