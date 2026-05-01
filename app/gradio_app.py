"""
Interactive web demo for the NLP Sentiment Suite.

Run with:
    python app/gradio_app.py

Then open http://localhost:7860 in your browser.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import SentimentModel, MODEL_REGISTRY
from src.pipeline import SentimentPipeline
from src.explainer import AttentionExplainer


# Load all models once at startup
print("Initializing models...")
pipeline = SentimentPipeline(model_keys=["distilbert", "roberta-twitter", "finbert"])
explainers = {
    key: AttentionExplainer(model)
    for key, model in pipeline.models.items()
}
print("All models ready.\n")


def run_prediction(text: str):
    """Called by Gradio on form submit. Returns a comparison table + attention plot."""
    if not text.strip():
        return "Please enter some text.", None

    comparison = pipeline.compare(text)

    # Format for Gradio Dataframe display
    display_df = comparison[["model", "label", "confidence", "latency_ms"]].copy()
    display_df.columns = ["Model", "Prediction", "Confidence", "Latency"]

    # Attention heatmap for distilbert (fastest)
    fig = explainers["distilbert"].explain(text)
    plt.close("all")

    return display_df, fig

def run_batch(file_obj):
    """Called when a CSV is uploaded. Runs all models and returns annotated CSV."""
    if file_obj is None:
        return None, "No file uploaded."

    try:
        df = pd.read_csv(file_obj.name)
        if "text" not in df.columns:
            return None, "CSV must have a 'text' column."

        df = pipeline.predict_dataframe(df, text_column="text")
        out_path = "/tmp/sentiment_results.csv"
        df.to_csv(out_path, index=False)
        return out_path, f"Done! Processed {len(df)} rows."
    except Exception as e:
        return None, f"Error: {str(e)}"
    
# Gradio UI
with gr.Blocks(title="NLP Sentiment Suite", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧠 NLP Sentiment Analysis Suite
    Compare three transformer models — **DistilBERT**, **RoBERTa-Twitter**, and **FinBERT** — on any text.
    Includes attention heatmap visualization for interpretability.
    """)

    with gr.Tab("Single Text"):
        with gr.Row():
            text_input = gr.Textbox(
                label="Input text",
                placeholder="e.g. 'The earnings report exceeded all expectations.'",
                lines=3,
            )
        submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Row():
            results_table = gr.Dataframe(label="Model Comparison", interactive=False)

        attention_plot = gr.Plot(label="Attention Heatmap (DistilBERT)")

        submit_btn.click(
            fn=run_prediction,
            inputs=[text_input],
            outputs=[results_table, attention_plot],
        )

        gr.Examples(
            examples=[
                ["Apple stock surged 8% after blowout Q3 earnings."],
                ["I'm absolutely devastated by the customer service experience."],
                ["The product is okay, nothing too special."],
                ["Breaking: Central bank raises rates for the fourth consecutive quarter."],
            ],
            inputs=text_input,
        )

    with gr.Tab("Batch CSV"):
        gr.Markdown("""
        Upload a CSV with a `text` column. All models will annotate every row.
        Download the results as a new CSV.
        """)
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        run_batch_btn = gr.Button("Run Batch Inference", variant="primary")
        batch_status = gr.Textbox(label="Status", interactive=False)
        file_output = gr.File(label="Download Results")

        run_batch_btn.click(
            fn=run_batch,
            inputs=[file_input],
            outputs=[file_output, batch_status],
        )

    with gr.Tab("Model Info"):
        model_info = []
        for key, config in MODEL_REGISTRY.items():
            model_info.append({
                "Key": key,
                "Model ID": config["model_id"],
                "Labels": ", ".join(config["labels"].values()),
                "Use case": config["description"],
            })
        gr.Dataframe(pd.DataFrame(model_info), interactive=False)


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
