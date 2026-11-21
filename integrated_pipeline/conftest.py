"""Test setup. Model integration tests are explicitly opt-in."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".venv", "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(_ROOT, ".venv", "models"))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")


def pytest_addoption(parser):
    parser.addoption(
        "--run-model-tests", action="store_true", default=False,
        help="Run integration tests with the real BGE-M3 encoder.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "model: requires the real BGE-M3 encoder")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-model-tests"):
        return
    skip = pytest.mark.skip(reason="Use --run-model-tests for real BGE-M3 integration tests")
    for item in items:
        if item.get_closest_marker("model"):
            item.add_marker(skip)


@pytest.fixture
def fake_semantic(monkeypatch):
    """Deterministic scores for API tests; makes no semantic-quality claim."""
    import task4_rank_evidence as t4

    def score(query, passages, model_config=None):
        return np.array([0.6 if (p.get("text") or "").strip() else 0.0 for p in passages])

    monkeypatch.setattr(t4, "semantic_relevance", score)
    return score
