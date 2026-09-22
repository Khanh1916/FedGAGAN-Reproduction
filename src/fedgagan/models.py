from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from fedgagan.config import ModelConfig, TrainConfig


def build_generator(output_dim: int, config: ModelConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(config.latent_dim,), name="noise")
    x = inputs
    for index, units in enumerate(config.generator_hidden):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"g_dense_{index + 1}")(x)
    outputs = tf.keras.layers.Dense(
        output_dim, activation=config.output_activation, name="synthetic_features"
    )(x)
    return tf.keras.Model(inputs, outputs, name="generator")


def build_critic(input_dim: int, config: ModelConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for index, units in enumerate(config.critic_hidden):
        x = tf.keras.layers.Dense(units, name=f"c_dense_{index + 1}")(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        if config.critic_dropout:
            x = tf.keras.layers.Dropout(config.critic_dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation=None, name="critic_score")(x)
    return tf.keras.Model(inputs, outputs, name="critic")


def gradient_penalty(
    critic: tf.keras.Model, real: tf.Tensor, fake: tf.Tensor
) -> tf.Tensor:
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform((batch_size, 1), 0.0, 1.0)
    interpolated = alpha * real + (1.0 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        prediction = critic(interpolated, training=True)
    gradients = tape.gradient(prediction, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-12)
    return tf.reduce_mean(tf.square(norm - 1.0))


@dataclass
class TrainHistory:
    critic_loss: list[float]
    generator_loss: list[float]
    gradient_penalty: list[float]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "critic_loss": self.critic_loss,
            "generator_loss": self.generator_loss,
            "gradient_penalty": self.gradient_penalty,
        }


class WGANTrainer:
    def __init__(
        self,
        generator: tf.keras.Model,
        critic: tf.keras.Model,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ):
        self.generator = generator
        self.critic = critic
        self.model_config = model_config
        self.train_config = train_config
        self.generator_optimizer = tf.keras.optimizers.Adam(
            train_config.learning_rate, beta_1=train_config.beta_1, beta_2=train_config.beta_2
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            train_config.learning_rate, beta_1=train_config.beta_1, beta_2=train_config.beta_2
        )

    @tf.function(reduce_retracing=True)
    def _critic_step(self, real: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        noise = tf.random.normal((tf.shape(real)[0], self.model_config.latent_dim))
        with tf.GradientTape() as tape:
            fake = self.generator(noise, training=True)
            real_score = self.critic(real, training=True)
            fake_score = self.critic(fake, training=True)
            gp = gradient_penalty(self.critic, real, fake)
            loss = (
                tf.reduce_mean(fake_score)
                - tf.reduce_mean(real_score)
                + self.train_config.gradient_penalty_weight * gp
            )
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        return loss, gp

    @tf.function(reduce_retracing=True)
    def _generator_step(self, batch_size: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal((batch_size, self.model_config.latent_dim))
        with tf.GradientTape() as tape:
            fake = self.generator(noise, training=True)
            loss = -tf.reduce_mean(self.critic(fake, training=True))
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )
        return loss

    def train(self, data: np.ndarray, epochs: int | None = None) -> TrainHistory:
        if len(data) < 2:
            raise ValueError("WGAN-GP requires at least two training rows")
        batch_size = min(self.train_config.batch_size, len(data))
        drop_remainder = len(data) >= batch_size
        dataset = (
            tf.data.Dataset.from_tensor_slices(data.astype(np.float32))
            .shuffle(len(data), reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=drop_remainder)
            .prefetch(tf.data.AUTOTUNE)
        )
        history = TrainHistory([], [], [])
        total_epochs = epochs if epochs is not None else self.train_config.epochs
        for _ in range(total_epochs):
            for step, real in enumerate(dataset):
                critic_loss = gp = None
                for _ in range(self.train_config.n_critic):
                    critic_loss, gp = self._critic_step(real)
                generator_loss = self._generator_step(tf.shape(real)[0])
                history.critic_loss.append(float(critic_loss.numpy()))
                history.generator_loss.append(float(generator_loss.numpy()))
                history.gradient_penalty.append(float(gp.numpy()))
                if (
                    self.train_config.steps_per_epoch
                    and step + 1 >= self.train_config.steps_per_epoch
                ):
                    break
        return history

    def train_steps(self, data: np.ndarray, steps: int) -> TrainHistory:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if len(data) < 2:
            raise ValueError("WGAN-GP requires at least two training rows")
        batch_size = min(self.train_config.batch_size, len(data))
        dataset = (
            tf.data.Dataset.from_tensor_slices(data.astype(np.float32))
            .shuffle(len(data), reshuffle_each_iteration=True)
            .repeat()
            .batch(batch_size, drop_remainder=True)
            .take(steps)
        )
        history = TrainHistory([], [], [])
        for real in dataset:
            critic_loss = gp = None
            for _ in range(self.train_config.n_critic):
                critic_loss, gp = self._critic_step(real)
            generator_loss = self._generator_step(tf.shape(real)[0])
            history.critic_loss.append(float(critic_loss.numpy()))
            history.generator_loss.append(float(generator_loss.numpy()))
            history.gradient_penalty.append(float(gp.numpy()))
        return history


def generate_samples(
    generator: tf.keras.Model, count: int, latent_dim: int, batch_size: int = 1024
) -> np.ndarray:
    outputs = []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        noise = tf.random.normal((size, latent_dim))
        outputs.append(generator(noise, training=False).numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)
