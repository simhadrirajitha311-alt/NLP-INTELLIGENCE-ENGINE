import numpy as np
import pandas as pd
import pytest

from nlp_intelligence_engine.data import summarize_text_dataset, validate_text_dataset
from nlp_intelligence_engine.features import NumpyTfidfVectorizer
from nlp_intelligence_engine.models import ClassicalTextClassifier
from nlp_intelligence_engine.preprocessing import tokenize_text


def test_tokenize_text_normalizes_case_and_punctuation():
    assert tokenize_text("Great, NLP isn't hard!") == ["great", "nlp", "isn", "t", "hard"]


def test_numpy_tfidf_is_normalized_and_ignores_unseen_terms():
    vectorizer = NumpyTfidfVectorizer().fit(["red apple", "green apple"])

    transformed = vectorizer.transform(["red apple", "blue"])

    assert transformed.shape == (2, 3)
    assert np.linalg.norm(transformed[0]) == pytest.approx(1.0)
    assert np.count_nonzero(transformed[1]) == 0


def test_numpy_tfidf_requires_fit():
    with pytest.raises(ValueError, match="fitted"):
        NumpyTfidfVectorizer().transform(["text"])


def test_dataset_validation_removes_exact_duplicates_and_reports_summary():
    frame = pd.DataFrame(
        {"text": [" good ", "good", "bad"], "label": ["positive", "positive", "negative"]}
    )

    cleaned = validate_text_dataset(frame, "text", "label")
    summary = summarize_text_dataset(cleaned, "text", "label")

    assert len(cleaned) == 2
    assert summary["class_counts"] == {"negative": 1, "positive": 1}
    assert summary["mean_text_length"] == 3.5


def test_dataset_validation_rejects_missing_labels():
    frame = pd.DataFrame({"text": ["hello"], "label": [None]})

    with pytest.raises(ValueError, match="missing values"):
        validate_text_dataset(frame, "text", "label")

    summary = summarize_text_dataset(frame, "text", "label")
    assert summary["missing_values"]["label"] == 1


def test_classifier_learns_vocabulary_from_training_texts():
    train_texts = ["bright helpful service", "kind support team", "awful delayed service", "rude support team"]
    train_labels = ["positive", "positive", "negative", "negative"]
    model = ClassicalTextClassifier().fit(train_texts, train_labels)

    prediction = model.predict(["bright service"])
    probabilities = model.predict_proba(["bright service"])
    learned_terms = model.pipeline.named_steps["tfidf"].vocabulary_

    assert prediction[0] == "positive"
    assert probabilities.shape == (1, 2)
    assert set(learned_terms) == set("bright helpful service kind support team awful delayed rude".split())