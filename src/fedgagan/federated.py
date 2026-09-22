from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from fedgagan.aggregation import weighted_average_weights
from fedgagan.config import FederatedConfig, ModelConfig, TrainConfig
from fedgagan.models import TrainHistory, WGANTrainer, build_critic, build_generator


@dataclass
class FederatedResult:
    generator: tf.keras.Model
    critic: tf.keras.Model
    history: list[dict[str, float | int | list[int]]]


class FederatedWGAN:
    def __init__(
        self,
        feature_dim: int,
        model_config: ModelConfig,
        train_config: TrainConfig,
        federated_config: FederatedConfig,
        seed: int,
    ):
        self.feature_dim = feature_dim
        self.model_config = model_config
        self.train_config = train_config
        self.federated_config = federated_config
        self.rng = np.random.default_rng(seed)
        self.global_generator = build_generator(feature_dim, model_config)
        self.global_critic = build_critic(feature_dim, model_config)

    def _local_train(
        self, client_data: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], TrainHistory]:
        generator = build_generator(self.feature_dim, self.model_config)
        critic = build_critic(self.feature_dim, self.model_config)
        generator.set_weights(self.global_generator.get_weights())
        critic.set_weights(self.global_critic.get_weights())
        trainer = WGANTrainer(
            generator, critic, self.model_config, copy.deepcopy(self.train_config)
        )
        local_epochs = self.federated_config.local_epochs
        if local_epochs is None:
            local_epochs = self.train_config.epochs
        history = trainer.train(client_data, epochs=local_epochs)
        return generator.get_weights(), critic.get_weights(), history

    def fit(self, clients: list[np.ndarray]) -> FederatedResult:
        if len(clients) != self.federated_config.num_clients:
            raise ValueError("Client partition count does not match federated.num_clients")
        log: list[dict[str, float | int | list[int]]] = []
        selected_count = max(
            1,
            math.ceil(self.federated_config.client_fraction * self.federated_config.num_clients),
        )
        for round_index in range(self.federated_config.rounds):
            selected = sorted(
                self.rng.choice(len(clients), selected_count, replace=False).tolist()
            )
            generator_updates = []
            critic_updates = []
            losses = []
            for client_index in selected:
                g_weights, c_weights, history = self._local_train(clients[client_index])
                size = len(clients[client_index])
                generator_updates.append((g_weights, size))
                critic_updates.append((c_weights, size))
                losses.extend(history.generator_loss)
            self.global_generator.set_weights(weighted_average_weights(generator_updates))
            self.global_critic.set_weights(weighted_average_weights(critic_updates))
            log.append(
                {
                    "round": round_index + 1,
                    "selected_clients": selected,
                    "generator_loss_mean": float(np.mean(losses)),
                }
            )
        return FederatedResult(self.global_generator, self.global_critic, log)
