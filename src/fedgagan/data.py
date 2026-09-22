from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fedgagan.config import DataConfig


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer

    @property
    def attack_train(self) -> np.ndarray:
        return self.x_train[self.y_train == 1]


def load_frames(config: DataConfig) -> pd.DataFrame:
    files: list[str] = []
    for pattern in config.paths:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        files.extend(matches)
    if not files:
        raise FileNotFoundError(f"No CSV files matched: {config.paths}")
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if config.max_rows and len(frame) > config.max_rows:
        frame = frame.sample(config.max_rows, random_state=42)
    return frame


def _binary_labels(series: pd.Series, attack_values: list[object]) -> np.ndarray:
    normalized = (
        series.astype(str).str.strip().str.lower().str.replace(r"\.0+$", "", regex=True)
    )
    attacks = {
        pd.Series([value])
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\.0+$", "", regex=True)
        .iloc[0]
        for value in attack_values
    }
    labels = normalized.isin(attacks).astype(np.int32).to_numpy()
    if len(np.unique(labels)) != 2:
        counts = series.value_counts(dropna=False).head(20).to_dict()
        raise ValueError(
            "Binary labels were not recovered. Check data.label_column and "
            f"data.attack_values. Observed labels: {counts}"
        )
    return labels


def prepare_data(config: DataConfig, seed: int) -> PreparedData:
    frame = load_frames(config)
    if config.label_column not in frame.columns:
        raise KeyError(
            f"Label column {config.label_column!r} missing. Available: {list(frame.columns)}"
        )
    y = _binary_labels(frame[config.label_column], config.attack_values)
    drop = [config.label_column, *config.drop_columns]
    features = frame.drop(columns=[column for column in drop if column in frame.columns])
    constant = [column for column in features if features[column].nunique(dropna=False) <= 1]
    features = features.drop(columns=constant)

    x_train_raw, x_test_raw, y_train_full, y_test = train_test_split(
        features,
        y,
        test_size=config.test_size,
        random_state=seed,
        stratify=y,
    )
    relative_validation = config.validation_size / (1.0 - config.test_size)
    if relative_validation > 0:
        x_train_raw, x_val_raw, y_train, y_val = train_test_split(
            x_train_raw,
            y_train_full,
            test_size=relative_validation,
            random_state=seed,
            stratify=y_train_full,
        )
    else:
        x_val_raw = x_train_raw.iloc[:0].copy()
        y_val = y_train_full[:0]
        y_train = y_train_full

    numeric = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    x_train = preprocessor.fit_transform(x_train_raw).astype(np.float32)
    x_val = preprocessor.transform(x_val_raw).astype(np.float32)
    x_test = preprocessor.transform(x_test_raw).astype(np.float32)
    names = preprocessor.get_feature_names_out().tolist()
    return PreparedData(x_train, y_train, x_val, y_val, x_test, y_test, names, preprocessor)


def save_prepared(data: PreparedData, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "splits.npz",
        x_train=data.x_train,
        y_train=data.y_train,
        x_validation=data.x_validation,
        y_validation=data.y_validation,
        x_test=data.x_test,
        y_test=data.y_test,
        feature_names=np.asarray(data.feature_names),
    )
    joblib.dump(data.preprocessor, directory / "preprocessor.joblib")


def load_prepared(directory: str | Path) -> PreparedData:
    directory = Path(directory)
    arrays = np.load(directory / "splits.npz", allow_pickle=False)
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    return PreparedData(
        arrays["x_train"],
        arrays["y_train"],
        arrays["x_validation"],
        arrays["y_validation"],
        arrays["x_test"],
        arrays["y_test"],
        arrays["feature_names"].astype(str).tolist(),
        preprocessor,
    )


def make_client_partitions(
    x: np.ndarray,
    num_clients: int,
    seed: int,
    method: str = "iid",
    alpha: float = 0.5,
) -> list[np.ndarray]:
    minimum_rows = 2
    if len(x) < minimum_rows * num_clients:
        raise ValueError("The attack subset must contain at least two rows per client")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    if method == "iid":
        return [x[part] for part in np.array_split(indices, num_clients)]
    if method != "dirichlet":
        raise ValueError(f"Unknown partition method: {method}")
    proportions = rng.dirichlet(np.full(num_clients, alpha))
    distributable = len(x) - minimum_rows * num_clients
    counts = np.floor(proportions * distributable).astype(int) + minimum_rows
    while counts.sum() > len(x):
        counts[np.argmax(counts)] -= 1
    while counts.sum() < len(x):
        counts[np.argmin(counts)] += 1
    offsets = np.cumsum([0, *counts])
    return [x[indices[offsets[i] : offsets[i + 1]]] for i in range(num_clients)]
