"""Classical NLP model implementations."""

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from nlp_intelligence_engine.preprocessing import tokenize_text


class ClassicalTextClassifier:
    """TF-IDF and Logistic Regression composed as one leakage-safe pipeline."""

    def __init__(self, max_iter: int = 1000) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(analyzer=tokenize_text)),
                ("classifier", LogisticRegression(max_iter=max_iter)),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "ClassicalTextClassifier":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: Sequence[str]):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: Sequence[str]):
        return self.pipeline.predict_proba(texts)