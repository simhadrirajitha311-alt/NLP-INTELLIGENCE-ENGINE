"""Educational feature extraction implemented with NumPy."""

from collections import Counter
from typing import Sequence

import numpy as np

from nlp_intelligence_engine.preprocessing import tokenize_text


class NumpyTfidfVectorizer:
    """A compact TF-IDF vectorizer for learning and small datasets."""

    def __init__(self) -> None:
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> "NumpyTfidfVectorizer":
        if len(texts) == 0:
            raise ValueError("texts must contain at least one document")

        tokenized = [tokenize_text(text) for text in texts]
        terms = sorted({term for document in tokenized for term in document})
        self.vocabulary_ = {term: index for index, term in enumerate(terms)}

        document_frequency = np.zeros(len(terms), dtype=np.float64)
        for document in tokenized:
            for term in set(document):
                document_frequency[self.vocabulary_[term]] += 1

        document_count = len(tokenized)
        self.idf_ = np.log((1 + document_count) / (1 + document_frequency)) + 1
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf_ is None:
            raise ValueError("vectorizer must be fitted before transform")

        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float64)
        for row, text in enumerate(texts):
            term_counts = Counter(tokenize_text(text))
            for term, count in term_counts.items():
                column = self.vocabulary_.get(term)
                if column is not None:
                    matrix[row, column] = count * self.idf_[column]

        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, row_norms, out=matrix, where=row_norms != 0)
        return matrix

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)