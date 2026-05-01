""" 
Shared preprocessing and helper utilities
"""

import re
import pandas as pd
from pathlib import Path
from typing import Optional


def clean_text(text: str) -> str:
    """ 
    Basic text normalization:
      - Strip leading/trailing whitespace
      - Collapse multiple spaces
      - Remove URLs
      - Remove HTML tags
    """
    text = text.strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_csv(path: str | Path, text_column: str, label_column = None) -> pd.DataFrame:
    """
    Load a CSV file and optionally validate required columns exist.

    :param path: path to the csv file
    :param text_column: column name containing input text
    :param label_column: column name containing ground truth-labels
    :return: cleaned dataframe
    """
    df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found! Available: {list(df.columns)}")
    
    if label_column and label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found!")
    
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    df = df[df[text_column].str.len() > 0].reset_index(drop=True) # Drops the empty rows and resets the indexing

    return df

def results_to_dataframe(results: list) -> pd.DataFrame:
    """
    Convert a list of PredictionResult objects into a flat DataFrame.
    """
    rows = []
    for r in results:
        row = {
            "text": r.text,
            "model": r.model_name,
            "label": r.label, 
            "confidence": r.confidence,
            "latency_ms": r.latency_ms,
        }
        row.update({f"score_{k}": v for k, v in r.all_scores.items()})
        rows.append(row)

    return pd.DataFrame(rows)

def truncate_text(text: str, max_chars: int = 80) -> str:
    """
    Truncate text for display purposes.
    """
    return text if len(text) <= max_chars else text[:max_chars] + "..."