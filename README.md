# NLP Intelligence Engine

A modular NLP platform built progressively, starting with an understandable and testable classical text-classification pipeline.

## Current Slice

The first vertical slice covers:

1. CSV dataset loading and basic validation with Pandas.
2. Lowercase, punctuation-aware tokenization.
3. A small NumPy TF-IDF implementation for learning and inspection.
4. A scikit-learn TF-IDF plus Logistic Regression pipeline for classification.
5. Tests for preprocessing, feature extraction, data validation, and train-to-predict behavior.

The production classifier fits its vocabulary as part of the training pipeline. Keep the test split separate and do not fit preprocessing or select models using test data.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

## Quick Example

```python
from nlp_intelligence_engine.models import ClassicalTextClassifier

texts = [
		"The support team solved my issue quickly",
		"My package is late and nobody has replied",
		"The agent was helpful and polite",
		"I am frustrated by this unresolved problem",
]
labels = ["positive", "negative", "positive", "negative"]

model = ClassicalTextClassifier().fit(texts, labels)
print(model.predict(["Helpful support, thank you"]))
```

## Learning Notes

- Tokenization is deliberately visible and small; it lowercases text and extracts word tokens without hiding behavior behind a large NLP toolkit.
- `NumpyTfidfVectorizer` demonstrates document frequency, smoothed inverse document frequency, and L2 normalization using NumPy arrays.
- The sklearn `Pipeline` is the serving-ready baseline. It keeps vectorizer fitting inside model training, which prevents vocabulary leakage from validation or test text.
- Pandas is used at the dataset boundary, where tabular validation and summaries are useful. NumPy is used for dense numerical feature calculations.

## Project Layout

```text
src/nlp_intelligence_engine/
	data.py            # CSV loading, validation, and dataset summary
	preprocessing.py   # transparent text tokenization
	features.py        # educational NumPy TF-IDF
	models.py          # sklearn classification pipeline
tests/               # focused unit and vertical-slice tests
```

## Roadmap

Build and understand one tested slice at a time: classical model comparison and evaluation, API serving, frontend, embeddings and retrieval, deep learning, transformers, then experiment tracking and deployment. The APIs and architecture will expand as each layer is introduced; future components are intentionally not scaffolded before they are useful.