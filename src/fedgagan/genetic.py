from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from fedgagan.config import GAConfig, TrainConfig


@dataclass
class Individual:
    learning_rate: float
    batch_size: int
    epochs: int
    beta_1: float
    beta_2: float
    n_critic: int
    gradient_penalty_weight: float
    score: float | None = None

    def to_train_config(self, base: TrainConfig) -> TrainConfig:
        result = copy.deepcopy(base)
        result.learning_rate = self.learning_rate
        result.batch_size = self.batch_size
        result.epochs = self.epochs
        result.beta_1 = self.beta_1
        result.beta_2 = self.beta_2
        result.n_critic = self.n_critic
        result.gradient_penalty_weight = self.gradient_penalty_weight
        return result

    def to_dict(self) -> dict:
        return asdict(self)


class GeneticSearch:
    """Roulette selection, one-point crossover, real mutation, and elitism."""

    genes = (
        "learning_rate",
        "batch_size",
        "epochs",
        "beta_1",
        "beta_2",
        "n_critic",
        "gradient_penalty_weight",
    )

    def __init__(self, config: GAConfig, seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _uniform(self, key: str) -> float:
        low, high = self.config.search_space[key]
        return float(self.rng.uniform(low, high))

    def _random_individual(self) -> Individual:
        batch_choices = self.config.search_space["batch_size"]
        epoch_low, epoch_high = self.config.search_space["epochs"]
        critic_low, critic_high = self.config.search_space["n_critic"]
        return Individual(
            learning_rate=10
            ** self.rng.uniform(
                np.log10(self.config.search_space["learning_rate"][0]),
                np.log10(self.config.search_space["learning_rate"][1]),
            ),
            batch_size=int(self.rng.choice(batch_choices)),
            epochs=int(self.rng.integers(epoch_low, epoch_high + 1)),
            beta_1=self._uniform("beta_1"),
            beta_2=self._uniform("beta_2"),
            n_critic=int(self.rng.integers(critic_low, critic_high + 1)),
            gradient_penalty_weight=self._uniform("gradient_penalty_weight"),
        )

    def _evaluate(self, population: list[Individual], objective: Callable[[Individual], float]):
        for individual in population:
            if individual.score is None:
                individual.score = float(objective(individual))

    def _select(self, population: list[Individual]) -> Individual:
        scores = np.asarray([item.score for item in population], dtype=float)
        fitness = 1.0 / (1.0 + scores - scores.min())
        probabilities = fitness / fitness.sum()
        return copy.deepcopy(population[int(self.rng.choice(len(population), p=probabilities))])

    def _crossover(self, left: Individual, right: Individual) -> tuple[Individual, Individual]:
        if self.rng.random() >= self.config.crossover_probability:
            return left, right
        point = int(self.rng.integers(1, len(self.genes)))
        left_values = [getattr(left, key) for key in self.genes]
        right_values = [getattr(right, key) for key in self.genes]
        child_a = dict(zip(self.genes, left_values[:point] + right_values[point:]))
        child_b = dict(zip(self.genes, right_values[:point] + left_values[point:]))
        return Individual(**child_a), Individual(**child_b)

    def _mutate(self, item: Individual) -> Individual:
        values = item.to_dict()
        values.pop("score", None)
        for key in self.genes:
            if self.rng.random() >= self.config.mutation_probability:
                continue
            if key == "batch_size":
                values[key] = int(self.rng.choice(self.config.search_space[key]))
            elif key in {"epochs", "n_critic"}:
                low, high = self.config.search_space[key]
                values[key] = int(self.rng.integers(low, high + 1))
            elif key == "learning_rate":
                low, high = self.config.search_space[key]
                values[key] = float(10 ** self.rng.uniform(np.log10(low), np.log10(high)))
            else:
                values[key] = self._uniform(key)
        return Individual(**values)

    def run(
        self, objective: Callable[[Individual], float]
    ) -> tuple[Individual, list[dict[str, float | int]]]:
        population = [self._random_individual() for _ in range(self.config.population_size)]
        history = []
        for generation in range(self.config.generations):
            self._evaluate(population, objective)
            population.sort(key=lambda item: item.score)
            history.append(
                {
                    "generation": generation + 1,
                    "best_kl": float(population[0].score),
                    "mean_kl": float(np.mean([item.score for item in population])),
                }
            )
            elites = [copy.deepcopy(item) for item in population[: self.config.elite_count]]
            children: list[Individual] = []
            while len(children) + len(elites) < self.config.population_size:
                first, second = self._crossover(
                    self._select(population), self._select(population)
                )
                children.extend([self._mutate(first), self._mutate(second)])
            population = elites + children[: self.config.population_size - len(elites)]
        self._evaluate(population, objective)
        best = min(population, key=lambda item: item.score)
        return best, history

