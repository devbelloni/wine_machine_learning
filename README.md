# Wine Classification: Red vs. White with Extra Trees

A supervised-learning study: classify wines as **red** or **white** from their physico-
chemical properties (acidity, sugar, chlorides, sulfur dioxide, density, pH, alcohol, ...),
using scikit-learn's `ExtraTreesClassifier`.

## Method

1. Load `wine_dataset.csv` and encode the target (`red` = 0, `white` = 1)
2. Split into training (70%) and test (30%) sets
3. Fit an `ExtraTreesClassifier`
4. Report test accuracy and inspect individual predictions

## Run it

```bash
pip install -r requirements.txt
python main.py
```

The script prints the shape of each split, the test accuracy and a few sample predictions
next to the true labels. The split is not seeded, so the exact accuracy varies slightly
between runs.

## Stack

Python, pandas, scikit-learn.

## Author

**Márcio Belloni**, AI/ML Engineer.
[LinkedIn](https://www.linkedin.com/in/marcio-belloni/) · [GitHub](https://github.com/devbelloni)
