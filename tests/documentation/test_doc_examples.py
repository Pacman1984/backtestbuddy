"""
Run every Python example in the README and the docs.

Blocks of one file run in order in a shared namespace, like a reader
copying them into a notebook. A run backtest is pre-seeded as
``backtest`` because reference pages assume one exists. Blocks that do
not compile (signature listings) are skipped. Plots are not displayed.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import backtestbuddy as btb

ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _seed_backtest():
    """A small run backtest for pages that assume one exists."""
    rng = np.random.default_rng(0)
    n = 120
    p = rng.uniform(0.3, 0.7, n)
    data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "odds_a": np.round(0.95 / p, 2),
        "odds_b": np.round(0.95 / (1 - p), 2),
        "winner": (rng.uniform(size=n) > p).astype(int),
        "prob_a": p,
        "prob_b": 1 - p,
        "prediction": (p < 0.5).astype(int),
    })
    backtest = btb.PredictionBacktest(
        data, ["odds_a", "odds_b"], "winner", "date", "prediction",
        model_prob_columns=["prob_a", "prob_b"], strategy=btb.FixedStake(10),
    )
    backtest.run()
    return backtest


def _runnable_blocks(path):
    blocks = []
    for index, source in enumerate(PYTHON_BLOCK.findall(path.read_text(encoding="utf-8"))):
        try:
            code = compile(source, f"{path.name}[block {index}]", "exec")
        except SyntaxError:
            continue  # signature listings, not meant to run
        blocks.append((index, code))
    return blocks


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_doc_examples_run(path, monkeypatch, tmp_path):
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)  # examples may write HTML files
    namespace = {"backtest": _seed_backtest()}
    for index, code in _runnable_blocks(path):
        try:
            exec(code, namespace)
        except Exception as error:  # noqa: BLE001 - report which block broke
            pytest.fail(f"{path.name} block {index} raised {type(error).__name__}: {error}")


def test_docs_contain_examples():
    """Guard against the extractor silently finding nothing."""
    total = sum(len(_runnable_blocks(path)) for path in DOC_FILES)
    assert total >= 20
