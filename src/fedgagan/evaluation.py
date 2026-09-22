from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from fedgagan.metrics import distribution_report


def _models(seed: int):
    return {
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed),
    }


def _score(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    prediction = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, probability)
    else:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": auc,
    }


def evaluate_classifiers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    synthetic_attacks: np.ndarray,
    seed: int,
) -> list[dict[str, str | float]]:
    benign = x_train[y_train == 0]
    synthetic_x = np.concatenate([benign, synthetic_attacks])
    synthetic_y = np.concatenate(
        [np.zeros(len(benign), dtype=int), np.ones(len(synthetic_attacks), dtype=int)]
    )
    hybrid_x = np.concatenate([x_train, synthetic_attacks])
    hybrid_y = np.concatenate([y_train, np.ones(len(synthetic_attacks), dtype=int)])
    scenarios = {
        "real": (x_train, y_train),
        "synthetic_attacks_plus_real_benign": (synthetic_x, synthetic_y),
        "hybrid": (hybrid_x, hybrid_y),
    }
    rows = []
    for scenario, (train_x, train_y) in scenarios.items():
        for name, model in _models(seed).items():
            model.fit(train_x, train_y)
            rows.append({"scenario": scenario, "classifier": name, **_score(model, x_test, y_test)})
    return rows


def create_plots(
    real: np.ndarray,
    fake: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = min(20, real.shape[1])
    index = np.arange(count)
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].bar(index - width / 2, real.mean(0)[:count], width, label="real")
    axes[0].bar(index + width / 2, fake.mean(0)[:count], width, label="synthetic")
    axes[0].set_title("Feature means")
    axes[1].bar(index - width / 2, real.std(0)[:count], width, label="real")
    axes[1].bar(index + width / 2, fake.std(0)[:count], width, label="synthetic")
    axes[1].set_title("Feature standard deviations")
    for axis in axes:
        axis.set_xticks(index, feature_names[:count], rotation=75, ha="right")
        axis.legend()
    fig.savefig(output / "means_and_stds.png", dpi=160)
    plt.close(fig)

    combined = np.concatenate([real, fake])
    projection = PCA(n_components=2, random_state=42).fit_transform(combined)
    labels = np.array(["real"] * len(real) + ["synthetic"] * len(fake))
    frame = pd.DataFrame({"PC1": projection[:, 0], "PC2": projection[:, 1], "source": labels})
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for source, subset in frame.groupby("source"):
        axis.scatter(subset.PC1, subset.PC2, s=8, alpha=0.45, label=source)
    axis.set_title("PCA: real attack vs synthetic attack")
    axis.legend()
    fig.savefig(output / "pca_real_vs_synthetic.png", dpi=160)
    plt.close(fig)


def full_evaluation(
    real_attacks: np.ndarray,
    synthetic_attacks: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    bins: int,
    seed: int,
    output_dir: str | Path,
) -> tuple[dict, list[dict]]:
    sample_count = min(len(real_attacks), len(synthetic_attacks), 10000)
    real = real_attacks[:sample_count]
    fake = synthetic_attacks[:sample_count]
    distribution = distribution_report(real, fake, bins)
    classifier_rows = evaluate_classifiers(
        x_train, y_train, x_test, y_test, synthetic_attacks, seed
    )
    create_plots(real, fake, feature_names, output_dir)
    return distribution, classifier_rows

