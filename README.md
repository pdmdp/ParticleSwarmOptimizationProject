# Particle swarm optimization

A small implementation of particle swarm optimization for bounded continuous functions. Each particle combines inertia with attraction to its personal best, a sampled neighborhood's best and the swarm's best position.

[Coursework notebook](FuzzySystems-EvolutionaryComputing-Project.ipynb) · [Experimental variant](pso_implementation.py) · [Assignment context](docs/assignment-context.md)

## Coursework specification and experimental variant

The **2024/25 submission** is the notebook linked above. The assignment explicitly prescribes `candidate - best` in its velocity equation (page 3), for the personal, neighborhood and global terms. The notebook follows that specified direction and uses the required `softpy` interfaces. Its saved sphere example reports fitness −1.508873 after 10,000 iterations.

The **November 2026 experiment** adds a separate NumPy implementation using the usual attractive direction, `best - position`. This changes the assignment's algorithm. It also uses a local random generator, restarts each fit, evaluates the final moved population and applies a different boundary rule. It does not implement the required `softpy` inheritance, so it is **not a replacement for the course submission**. The benchmarks below concern this later variant only. [The distinction is documented here](docs/assignment-context.md).

## Run the experimental variant

```bash
git clone https://github.com/pdmdp/ParticleSwarmOptimizationProject.git
cd ParticleSwarmOptimizationProject
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python benchmark.py
```

```python
import numpy as np
from pso_implementation import ParticleSwarmOptimizer

optimizer = ParticleSwarmOptimizer(
    fitness_func=lambda particle: -np.sum(particle.candidate**2),
    pop_size=15,
    n_neighbors=3,
    size=2,
    lower=-10.0,
    upper=10.0,
    seed=42,
)
best = optimizer.fit(n_iters=300)
print(best.candidate, optimizer.global_fitness_best)
```

The interface **maximizes** fitness. Negate a loss to minimize it. Scalar bounds are broadcast to all dimensions; arrays allow a separate interval for each dimension. `n_iters` counts position updates, so the objective is evaluated on the initial population and after every update. Repeating `fit` with an integer seed starts the same experiment again.

## Seeded checks

![Median best loss and interquartile range across ten seeds for three functions](assets/convergence.png)

All checks use 15 particles, three sampled neighbors, 300 updates, bounds `[−5.12, 5.12]` in both dimensions, and seeds 0–9. The shaded range is the interquartile range. Losses below `1e-16` are clipped to that value for the logarithmic plot only.

| Function | Median final loss | Best | Worst |
| :--- | ---: | ---: | ---: |
| Sphere, 2D | 1.86e−43 | 3.97e−44 | 8.79e−43 |
| Rosenbrock, 2D | 0.00747 | 0.00000 | 0.62592 |
| Rastrigin, 2D | 0.99496 | 0.00000 | 0.99496 |

The sphere converges reliably in these runs. The other two functions show seed dependence and local optima. These are small correctness and behavior checks, not evidence of general superiority over other optimizers. Full traces and settings are in [results/benchmark.json](results/benchmark.json).

Six tests cover attraction direction, convergence, bounds and final-step evaluation, repeated fits, single-particle/empty-neighborhood behavior, and invalid inputs. The variant was tested with Python 3.14.4 and NumPy 2.4.3. The original notebook's separate dependencies are in `requirements-original.txt`.

## Update rule

With independent scalar random coefficients drawn uniformly from `[0, 1)`, the experimental variant uses:

```text
v = 0.7 v
  + 0.1 r_personal (personal_best - position)
  + 0.1 r_neighbor (neighborhood_best - position)
  + 0.8 r_global   (global_best - position)
```

Positions are clipped to their bounds and outward velocity is set to zero on clipped dimensions. The weights retain the coursework configuration; they are not a tuned recommendation for arbitrary objectives.

**Matteo Pietro Di Pilato** · Fuzzy Systems and Evolutionary Computing, BSc in Artificial Intelligence. Code: [MIT](LICENSE).
