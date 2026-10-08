"""Pandas-based dataset loading and lightweight quality checks."""

from pathlib import Path

import pandas as pd


def validate_text_dataset(
    frame: pd.DataFrame, text_column: str, label_column: str
) -> pd.DataFrame:
    """Return a clean copy or raise when required training data is invalid."""
    missing_columns = {text_column, label_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    dataset = frame[[text_column, label_column]].copy()
    if dataset[[text_column, label_column]].isna().any().any():
        raise ValueError("text and label columns must not contain missing values")

    dataset[text_column] = dataset[text_column].astype(str).str.strip()
    if dataset[text_column].eq("").any():
        raise ValueError("text values must not be empty")

    return dataset.drop_duplicates().reset_index(drop=True)


def load_text_dataset(
    path: str | Path, text_column: str, label_column: str
) -> pd.DataFrame:
    """Load a CSV and validate its text and label columns."""
    frame = pd.read_csv(path)
    return validate_text_dataset(frame, text_column, label_column)


def summarize_text_dataset(
    frame: pd.DataFrame, text_column: str, label_column: str
) -> dict[str, object]:
    """Summarize raw data quality, class balance, and text lengths."""
    missing_columns = {text_column, label_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    class_counts = frame[label_column].dropna().value_counts().sort_index()
    available_text = frame[text_column].dropna().astype(str).str.strip()
    text_lengths = available_text.str.len()
    return {
        "row_count": len(frame),
        "class_counts": class_counts.to_dict(),
        "missing_values": frame[[text_column, label_column]].isna().sum().to_dict(),
        "mean_text_length": float(text_lengths.mean()) if len(text_lengths) else 0.0,
    }