from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str = "demo"
    paths: list[str] = field(default_factory=lambda: ["data/raw/demo.csv"])
    label_column: str = "label"
    attack_values: list[Any] = field(default_factory=lambda: [1, "1", "attack"])
    drop_columns: list[str] = field(default_factory=list)
    max_rows: int | None = None
    test_size: float = 0.2
    validation_size: float = 0.1


@dataclass
class ModelConfig:
    latent_dim: int = 32
    generator_hidden: list[int] = field(default_factory=lambda: [128, 512, 512, 128])
    critic_hidden: list[int] = field(default_factory=lambda: [128, 512, 512, 128])
    critic_dropout: float = 0.2
    output_activation: str = "sigmoid"


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    beta_1: float = 0.5
    beta_2: float = 0.9
    batch_size: int = 64
    epochs: int = 100
    steps_per_epoch: int | None = None
    n_critic: int = 5
    gradient_penalty_weight: float = 10.0


@dataclass
class FederatedConfig:
    num_clients: int = 5
    client_fraction: float = 1.0
    rounds: int = 10
    local_epochs: int | None = 1
    partition: str = "iid"
    dirichlet_alpha: float = 0.5


@dataclass
class GAConfig:
    enabled: bool = True
    population_size: int = 12
    generations: int = 5
    crossover_probability: float = 0.6
    mutation_probability: float = 0.1
    evaluation_steps: int = 30
    elite_count: int = 1
    search_space: dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": [1e-5, 1e-3],
            "batch_size": [32, 64, 128, 256, 512],
            "epochs": [1000, 50000],
            "beta_1": [0.5, 0.9],
            "beta_2": [0.9, 0.999],
            "n_critic": [1, 10],
            "gradient_penalty_weight": [1.0, 20.0],
        }
    )


@dataclass
class EvaluationConfig:
    synthetic_samples: int = 1000
    histogram_bins: int = 20
    classifier_random_state: int = 42


@dataclass
class ProjectConfig:
    seed: int = 42
    output_dir: str = "artifacts/demo"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, raw: dict[str, Any] | None):
    return cls(**(raw or {}))


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = ProjectConfig(
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "artifacts/demo"),
        data=_construct(DataConfig, raw.get("data")),
        model=_construct(ModelConfig, raw.get("model")),
        train=_construct(TrainConfig, raw.get("train")),
        federated=_construct(FederatedConfig, raw.get("federated")),
        ga=_construct(GAConfig, raw.get("ga")),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation")),
    )
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    cfg.data.paths = [str((base / item).resolve()) for item in cfg.data.paths]
    cfg.output_dir = str((base / cfg.output_dir).resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg: ProjectConfig) -> None:
    if not 0 < cfg.data.test_size < 1:
        raise ValueError("data.test_size must be between 0 and 1")
    if not 0 <= cfg.data.validation_size < 1:
        raise ValueError("data.validation_size must be in [0, 1)")
    if cfg.data.test_size + cfg.data.validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")
    if cfg.model.latent_dim <= 0:
        raise ValueError("model.latent_dim must be positive")
    if cfg.train.n_critic <= 0 or cfg.train.batch_size <= 1:
        raise ValueError("n_critic must be positive and batch_size must exceed 1")
    if cfg.federated.num_clients <= 0 or not 0 < cfg.federated.client_fraction <= 1:
        raise ValueError("Invalid federated client settings")
    if cfg.federated.local_epochs is not None and cfg.federated.local_epochs <= 0:
        raise ValueError("federated.local_epochs must be positive or null")
    if cfg.federated.partition not in {"iid", "dirichlet"}:
        raise ValueError("federated.partition must be iid or dirichlet")
