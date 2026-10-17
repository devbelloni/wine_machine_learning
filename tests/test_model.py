from pathlib import Path

import pandas as pd
import pytest

from wine_classifier.cli import main
from wine_classifier.model import LABELS, evaluate, load_dataset

DATA = Path(__file__).resolve().parents[1] / "data" / "wine_dataset.csv"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA)


def test_load_dataset_encodes_target_and_drops_style_from_features(dataset):
    x, y = dataset
    assert "style" not in x.columns
    assert set(y.unique()) == set(LABELS.values())
    assert len(x) == len(y)


def test_load_dataset_rejects_unknown_labels(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2], "style": ["red", "rose"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Unexpected values"):
        load_dataset(bad)


def test_evaluate_is_reproducible_and_accurate(dataset):
    x, y = dataset
    _, a = evaluate(x, y, seed=7)
    _, b = evaluate(x, y, seed=7)
    assert a.accuracy == b.accuracy
    assert a.accuracy > 0.97  # red vs. white wine is an easy, well-separated problem
    assert sum(sum(row) for row in a.confusion) == round(len(x) * 0.3)


def test_train_and_predict_cli(tmp_path, dataset):
    out = tmp_path / "models"
    assert main(["train", "--data", str(DATA), "--out", str(out)]) == 0
    assert (out / "model.joblib").exists() and (out / "metrics.json").exists()

    x, _ = dataset
    sample = tmp_path / "sample.csv"
    x.head(20).to_csv(sample, index=False)
    result = tmp_path / "result.csv"
    assert main(["predict", "--model", str(out / "model.joblib"), "--csv", str(sample), "--output", str(result)]) == 0
    assert "predicted_style" in pd.read_csv(result).columns
