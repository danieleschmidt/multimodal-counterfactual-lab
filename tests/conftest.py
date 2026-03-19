"""Shared fixtures for counterfactual-lab tests."""

import pytest


@pytest.fixture
def simple_classifier():
    """A deterministic dummy classifier for testing."""
    def predict(text: str) -> dict:
        text_lower = text.lower()
        if any(w in text_lower for w in ["excellent", "outstanding", "great", "brilliant"]):
            return {"label": "POSITIVE", "score": 0.9}
        if any(w in text_lower for w in ["terrible", "horrible", "bad"]):
            return {"label": "NEGATIVE", "score": 0.1}
        return {"label": "NEUTRAL", "score": 0.5}
    return predict


@pytest.fixture
def biased_classifier():
    """A classifier that's biased: always labels male-gendered text as POSITIVE."""
    def predict(text: str) -> dict:
        text_lower = text.lower()
        # Biased: he/him/his → always POSITIVE
        if any(w in text_lower.split() for w in ["he", "him", "his", "man", "male"]):
            return {"label": "POSITIVE", "score": 0.85}
        return {"label": "NEGATIVE", "score": 0.3}
    return predict
