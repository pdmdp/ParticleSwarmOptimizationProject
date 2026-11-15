"""Seeded PSO for bounded maximization, reviewed in November 2026.

The submitted softpy notebook is retained separately. This implementation uses
NumPy directly and keeps the fitness_func(particle) and fit(n_iters) interface.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass
class ParticleCandidate:
    candidate: np.ndarray
    velocity: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def copy(self):
        return ParticleCandidate(*(a.copy() for a in
                                   (self.candidate, self.velocity, self.lower, self.upper)))

    def recombine(self, personal_best, neighborhood_best, global_best, rng):
        """Update velocity toward the three remembered best positions."""
        r_personal, r_neighbor, r_global = rng.random(3)
        self.velocity = (
            0.7 * self.velocity
            + 0.1 * r_personal * (personal_best.candidate - self.candidate)
            + 0.1 * r_neighbor * (neighborhood_best.candidate - self.candidate)
            + 0.8 * r_global * (global_best.candidate - self.candidate)
        )

    def mutate(self):
        """Move, clamp to the search box and stop outward boundary velocity."""
        proposed = self.candidate + self.velocity
        clipped = (proposed < self.lower) | (proposed > self.upper)
        self.candidate = np.clip(proposed, self.lower, self.upper)
        self.velocity[clipped] = 0.0


class ParticleSwarmOptimizer:
    """Maximize a finite scalar fitness inside a box.

    Each fit starts a fresh swarm. An integer seed makes repeated fits repeatable.
    n_iters counts position updates; the initial and final populations are evaluated.
    Neighbors are sampled without replacement at each update. With no neighbors,
    the particle's own best position supplies the neighborhood term.
    """

    def __init__(self, fitness_func, pop_size=15, n_neighbors=3, *, size=2,
                 lower=-10.0, upper=10.0, seed=None):
        if not callable(fitness_func):
            raise TypeError("fitness_func must be callable")
        self.fitness_func = fitness_func
        self.pop_size = _integer(pop_size, "pop_size", 1)
        self.n_neighbors = _integer(n_neighbors, "n_neighbors", 0)
        self.size = _integer(size, "size", 1)
        try:
            self.lower = np.broadcast_to(np.asarray(lower, dtype=float), (self.size,)).copy()
            self.upper = np.broadcast_to(np.asarray(upper, dtype=float), (self.size,)).copy()
        except ValueError as exc:
            raise ValueError("bounds must be scalar or have one value per dimension") from exc
        if not (np.isfinite(self.lower).all() and np.isfinite(self.upper).all()
                and np.all(self.lower < self.upper)
                and np.isfinite(self.upper - self.lower).all()):
            raise ValueError("bounds must be finite, with lower < upper in each dimension")
        self.seed = None if seed is None else _integer(seed, "seed", 0)
        self.population = []
        self.global_best = None
        self.global_fitness_best = -np.inf
        self.history_ = []

    def _evaluate(self):
        for i, particle in enumerate(self.population):
            score = float(self.fitness_func(particle))
            if not np.isfinite(score):
                raise ValueError("fitness_func must return a finite scalar")
            if score > self.fitness_best[i]:
                self.fitness_best[i] = score
                self.best[i] = particle.copy()
            if score > self.global_fitness_best:
                self.global_fitness_best = score
                self.global_best = particle.copy()
        self.history_.append(self.global_fitness_best)

    def fit(self, n_iters=200):
        n_iters = _integer(n_iters, "n_iters", 0)
        rng = np.random.default_rng(self.seed)
        width = self.upper - self.lower
        self.population = [ParticleCandidate(
            rng.uniform(self.lower, self.upper), rng.uniform(-width, width),
            self.lower.copy(), self.upper.copy(),
        ) for _ in range(self.pop_size)]
        self.best = [p.copy() for p in self.population]
        self.fitness_best = np.full(self.pop_size, -np.inf)
        self.global_best = None
        self.global_fitness_best = -np.inf
        self.history_ = []
        self._evaluate()
        for _ in range(n_iters):
            # Best states stay fixed while the entire swarm moves.
            for i, particle in enumerate(self.population):
                others = np.delete(np.arange(self.pop_size), i)
                count = min(self.n_neighbors, len(others))
                if count:
                    neighbors = rng.choice(others, size=count, replace=False)
                    best_neighbor = self.best[neighbors[np.argmax(self.fitness_best[neighbors])]]
                else:
                    best_neighbor = self.best[i]
                particle.recombine(self.best[i], best_neighbor, self.global_best, rng)
                particle.mutate()
            self._evaluate()
        return self.global_best.copy()
