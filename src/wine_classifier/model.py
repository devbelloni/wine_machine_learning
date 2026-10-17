"""Data loading, training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

TARGET = "style"
LABELS = {"red": 0, "white": 1}


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and the encoded target (red = 0, white = 1)."""
    df = pd.read_csv(path)
    if TARGET not in df:
        raise ValueError(f"Column '{TARGET}' not found in {path}")
    unknown = set(df[TARGET].unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected values in '{TARGET}': {sorted(unknown)}")
    df = df.drop_duplicates().reset_index(drop=True)
    return df.drop(columns=[TARGET]), df[TARGET].map(LABELS)


def build_model(seed: int = 42, n_estimators: int = 300) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


@dataclass
class Evaluation:
    accuracy: float
    f1_macro: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion: list[list[int]]
    report: str
    feature_importance: dict[str, float]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate(x: pd.DataFrame, y: pd.Series, seed: int = 42, test_size: float = 0.3) -> tuple[object, Evaluation]:
    """Stratified hold-out evaluation plus 5-fold stratified cross-validation."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = build_model(seed).fit(x_train, y_train)
    pred = model.predict(x_test)

    cv = cross_val_score(
        build_model(seed), x, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="accuracy"
    )
    importance = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    result = Evaluation(
        accuracy=float(accuracy_score(y_test, pred)),
        f1_macro=float(f1_score(y_test, pred, average="macro")),
        cv_accuracy_mean=float(cv.mean()),
        cv_accuracy_std=float(cv.std()),
        confusion=confusion_matrix(y_test, pred).tolist(),
        report=classification_report(y_test, pred, target_names=list(LABELS)),
        feature_importance={k: round(float(v), 4) for k, v in importance.items()},
    )
    return model, result
