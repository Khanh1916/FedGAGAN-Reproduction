from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from fedgagan.config import ProjectConfig, load_config
from fedgagan.data import (
    PreparedData,
    load_prepared,
    make_client_partitions,
    prepare_data,
    save_prepared,
)
from fedgagan.evaluation import full_evaluation
from fedgagan.federated import FederatedWGAN
from fedgagan.genetic import GeneticSearch, Individual
from fedgagan.metrics import histogram_kl
from fedgagan.models import WGANTrainer, build_critic, build_generator, generate_samples
from fedgagan.utils import ensure_dir, set_global_determinism, write_json


class Pipeline:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.output = ensure_dir(config.output_dir)
        set_global_determinism(config.seed)
        with (self.output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    def preprocess(self) -> PreparedData:
        data = prepare_data(self.config.data, self.config.seed)
        save_prepared(data, self.output / "prepared")
        write_json(
            self.output / "data_summary.json",
            {
                "train_rows": len(data.x_train),
                "validation_rows": len(data.x_validation),
                "test_rows": len(data.x_test),
                "feature_count": len(data.feature_names),
                "train_attack_rows": int(data.y_train.sum()),
                "train_benign_rows": int((data.y_train == 0).sum()),
                "feature_names": data.feature_names,
            },
        )
        return data

    def _data(self) -> PreparedData:
        if (self.output / "prepared" / "splits.npz").exists():
            return load_prepared(self.output / "prepared")
        return self.preprocess()

    def optimize(self, data: PreparedData | None = None) -> Individual | None:
        if not self.config.ga.enabled:
            return None
        data = data or self._data()
        real = data.attack_train

        def objective(individual: Individual) -> float:
            tf.keras.backend.clear_session()
            train_config = individual.to_train_config(self.config.train)
            generator = build_generator(real.shape[1], self.config.model)
            critic = build_critic(real.shape[1], self.config.model)
            trainer = WGANTrainer(generator, critic, self.config.model, train_config)
            trainer.train_steps(real, self.config.ga.evaluation_steps)
            count = min(len(real), 2048)
            fake = generate_samples(generator, count, self.config.model.latent_dim)
            return histogram_kl(real[:count], fake, self.config.evaluation.histogram_bins)

        best, history = GeneticSearch(self.config.ga, self.config.seed).run(objective)
        write_json(self.output / "ga_best.json", best.to_dict())
        write_json(self.output / "ga_history.json", history)
        return best

    def _selected_train_config(self):
        best_path = self.output / "ga_best.json"
        if best_path.exists():
            with best_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            raw.pop("score", None)
            return Individual(**raw).to_train_config(self.config.train)
        return copy.deepcopy(self.config.train)

    def train(self, data: PreparedData | None = None):
        data = data or self._data()
        train_config = self._selected_train_config()
        clients = make_client_partitions(
            data.attack_train,
            self.config.federated.num_clients,
            self.config.seed,
            self.config.federated.partition,
            self.config.federated.dirichlet_alpha,
        )
        federation = FederatedWGAN(
            data.x_train.shape[1],
            self.config.model,
            train_config,
            self.config.federated,
            self.config.seed,
        )
        result = federation.fit(clients)
        model_dir = ensure_dir(self.output / "models")
        result.generator.save(model_dir / "generator.keras")
        result.critic.save(model_dir / "critic.keras")
        write_json(self.output / "federated_history.json", result.history)
        return result.generator

    def _generator(self):
        path = self.output / "models" / "generator.keras"
        if not path.exists():
            return self.train()
        return tf.keras.models.load_model(path)

    def generate(self, generator=None) -> np.ndarray:
        if generator is None:
            generator = self._generator()
        fake = generate_samples(
            generator,
            self.config.evaluation.synthetic_samples,
            self.config.model.latent_dim,
        )
        np.save(self.output / "synthetic_attacks.npy", fake)
        data = self._data()
        pd.DataFrame(fake, columns=data.feature_names).to_csv(
            self.output / "synthetic_attacks_encoded.csv", index=False
        )
        return fake

    def evaluate(self, data: PreparedData | None = None, fake: np.ndarray | None = None):
        data = data or self._data()
        if fake is None:
            synthetic_path = self.output / "synthetic_attacks.npy"
            fake = np.load(synthetic_path) if synthetic_path.exists() else self.generate()
        distribution, classifiers = full_evaluation(
            data.attack_train,
            fake,
            data.x_train,
            data.y_train,
            data.x_test,
            data.y_test,
            data.feature_names,
            self.config.evaluation.histogram_bins,
            self.config.evaluation.classifier_random_state,
            self.output / "plots",
        )
        write_json(self.output / "distribution_metrics.json", distribution)
        pd.DataFrame(classifiers).to_csv(self.output / "classifier_metrics.csv", index=False)
        return distribution, classifiers

    def run(self, stages: list[str]) -> None:
        data = None
        generator = None
        fake = None
        for stage in stages:
            if stage == "preprocess":
                data = self.preprocess()
            elif stage == "search":
                self.optimize(data)
            elif stage == "train":
                generator = self.train(data)
            elif stage == "generate":
                fake = self.generate(generator)
            elif stage == "evaluate":
                self.evaluate(data, fake)
            else:
                raise ValueError(f"Unknown pipeline stage: {stage}")


def run_from_path(config_path: str | Path, stages: list[str]) -> None:
    Pipeline(load_config(config_path)).run(stages)
