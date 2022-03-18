import abc
import itertools
import math
import random
from typing import *

import datasets
import pandas as pd
import numpy as np
import numpy.typing as npt
import scipy.special
import rich
import torch
import transformers


def batchify(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]


###############################################################################
# Moving averages
###############################################################################
class FixedLengthRollingAverage:
    def __init__(self, window_size: int, default: Optional[float] = None):
        self.window_size = window_size
        self.seen_full = False
        self.window = np.zeros(window_size)
        self.index = 0
        self.default = default

    def push(self, x: float) -> float:
        self.window[self.index] = x
        self.index += 1
        if self.index == self.window_size:
            self.seen_full = True
        assert not self.index > self.window_size, (self.index, self.window_size)
        self.index %= self.window_size
        return self.get()

    def get(self) -> float:
        if not self.seen_full and self.index == 0 and self.default is not None:
            return self.default

        if self.seen_full:
            return self.window.mean()
        else:
            return self.window[:self.index].mean()


class EMA:
    def __init__(self, alpha: float, default: Optional[float] = None):
        self.alpha = alpha
        self.value = default

    def push(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * x
        return self.value
    
    def get(self) -> Optional[float]:
        return self.value


###############################################################################
# Non Stationary Multi-Armed Bandits
###############################################################################
class MAB:
    def __init__(self, num_arms: int, average_class, average_class_args):
        self.num_arms = num_arms
        self.rewards = [average_class(**average_class_args) for _ in range(num_arms)]

    def update(self, index, reward):
        self.rewards[index].push(reward)

    def get(self):
        pass


class NSEpsilonGreedy(MAB):
    def __init__(self, num_arms: int, average_class, average_class_args, epsilon: float):
        super().__init__(num_arms, average_class, average_class_args)
        self.epsilon = epsilon

    def get(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax([r.get() for r in self.rewards])


class OurMAB(MAB):
    def __init__(
        self, 
        num_arms: int, 
        average_class, 
        average_class_args, 
        epsilon: float, 
        temperature: float = 1.
    ):
        super().__init__(num_arms, average_class, average_class_args)
        
        self.epsilon = epsilon
        self.temperature = temperature

    def get(self):
        prob = (
            scipy.special.softmax(
                [x.get() / self.temperature for x in self.rewards]
            ) * (1 - self.epsilon) + (self.epsilon) / self.num_arms
        )
        return np.random.multinomial(1, prob).argmax()
        

class Exp3:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self.gamma = 1 / num_arms
        self.weights = np.ones(self.num_arms)
        self.probability_distribution = None

    def get(self) -> int:
        self.probability_distribution = self._distr(self.weights, self.gamma)
        return np.random.multinomial(1, self.probability_distribution).argmax()

    def update(self, index: int, reward: float) -> None:
        estimated_reward = reward / self.probability_distribution[index]
        self.weights[index] *= np.exp(estimated_reward * self.gamma / self.num_arms)        

    @classmethod
    def _distr(
        cls, 
        weights: npt.NDArray[np.float64], 
        gamma: float
    ) -> npt.NDArray[np.float64]:

        sum_of_weights = np.sum(weights)
        return (1.0 - gamma) * (weights / sum_of_weights) + (gamma / len(weights))


class Exp3S:
    def __init__(self):
        raise NotImplementedError


###############################################################################
# Linear Curriculum
###############################################################################
class LinearCurriculum:
    def __init__(self, num_arms, max_num_steps):
        self.num_args = num_arms
        self.num_steps = max_num_steps
        self.step_position = 0

    def get(self) -> float:
        if self.step_position // self.num_steps < self.num_steps:
            return self.step_position // self.num_steps
        else:
            return random.randint(0, self.num_steps - 1)

    def update(self, *args, **kwargs) -> None:
        self.step_position += 1


###############################################################################
# AutoCurriculums
###############################################################################
class LengthAutoCurriculumDL:
    # This is the super dump implementation of the auto curriculum, 
    # using EMA and Softmax

    def __init__(
        self, 
        dataset, 
        field_name, 
        num_clusters, 
        batch_size, 
        cluster_picker, 
        collator
    ):
        self.index = None
        dataset = dataset.map(lambda x: {f"len_{field_name}" : len(x[field_name])})
        sort_col_name = f"len_{field_name}"
        dataset = dataset.sort(sort_col_name)
        dataset = dataset.remove_columns(sort_col_name)
        self.field_name = field_name
        self.batch_size = batch_size
        self.cluster_size = math.ceil(len(dataset) / num_clusters)
        self.clusters = [
            pd.DataFrame(
                dataset[i * self.cluster_size: (i + 1) * self.cluster_size]
            ).to_dict("records") 
            for i in range(num_clusters)
        ]
        self.cluster_picker = cluster_picker
        self.already_running = False
        self.collator = collator

    def __iter__(self):
        assert not self.already_running, "Already running"
        self.already_running = True
        return self.Iterator(
            self.cluster_picker, 
            self.clusters, 
            batch_size=self.batch_size, 
            collator=self.collator,
        )
    
    class Iterator:
        def __init__(self, cluster_picker, clusters, batch_size, collator):
            self.index = 0
            self.cluster_picker = cluster_picker
            self.clusters = clusters
            self.batch_size = batch_size
            self.batchers = [batchify(cluster, batch_size) for cluster in clusters]
            self.collator = collator

        def __iter__(self):
            return self

        def __next__(self):
            cluster_chosen = self.cluster_picker.get()

            try:
                batch = next(self.batchers[cluster_chosen]) 
            except StopIteration:
                self.batchers[cluster_chosen] = batchify(
                    self.clusters[cluster_chosen], self.batch_size
                )
                batch = next(self.batchers[cluster_chosen]) 

            rich.print(f"[green]cluster {cluster_chosen + 1} / {len(self.clusters)}")
            return cluster_chosen, self.collator(batch)

        def update(self, index, reward):
            self.cluster_picker.update(index, reward)

    def update_pg(self, arm_idx: int, pg: float):
        self.cluster_picker.update(arm_idx, pg)


