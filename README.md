# Wine Classification: Red vs. White

![CI](https://github.com/devbelloni/wine_machine_learning/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A supervised-learning project that classifies wines as **red** or **white** from their
physico-chemical properties (acidity, sugar, chlorides, sulfur dioxide, density, pH,
alcohol, and so on) with an Extra Trees classifier. The code is a tested package with a
command-line interface, a stratified evaluation protocol and a saved model.

## Results (seed 42)

| Metric | Value |
|---|---|
| Hold-out accuracy (30%, stratified, 1,596 wines) | **99.69%** |
| 5-fold stratified CV accuracy | **99.53% ± 0.27%** |
| Macro F1 (hold-out) | 0.997 |

Confusion matrix on the hold-out set:

|  | predicted red | predicted white |
|---|---|---|
| **actual red** | 403 | 5 |
| **actual white** | 0 | 1,188 |

Most important features: `total_sulfur_dioxide` (0.29), `volatile_acidity` (0.15),
`chlorides` (0.11), `density` (0.10), `fixed_acidity` (0.07).

Red vs. white is a well-separated problem, so near-perfect accuracy is expected. The value
of the project is the methodology, not the number:

- **Duplicates removed** before splitting (the raw file has 6,497 rows; 5,320 are unique), so
  identical wines cannot leak from the training set into the test set.
- **Stratified split and CV**, because the classes are imbalanced (about 3:1 white to red).
- **Cross-validation next to the hold-out score**, so the estimate does not depend on a
  single lucky split.
- **Fixed seeds** everywhere for reproducibility.

## Usage

```bash
pip install -e ".[dev]"

wine-classifier train                                  # trains, evaluates, saves models/
wine-classifier predict --csv new_wines.csv --output predictions.csv
```

`train` writes `models/model.joblib` and `models/metrics.json`. `predict` needs a CSV with
the same feature columns as `data/wine_dataset.csv` and adds a `predicted_style` column.

## Data

`data/wine_dataset.csv` has 11 physico-chemical features, a `quality` score and the `style`
label (`red` or `white`). It is based on the public Wine Quality data (Cortez et al., 2009).

## Development

```bash
pytest -q && ruff check .
```

The tests cover data validation, reproducibility, accuracy and the train/predict CLI.

## Project layout

```
src/wine_classifier/   model.py (data, training, evaluation) · cli.py
tests/                 pytest suite
data/                  wine_dataset.csv
```

## Author

**Márcio Belloni**, AI/ML Engineer.
[LinkedIn](https://www.linkedin.com/in/marcio-belloni/) · [GitHub](https://github.com/devbelloni)
