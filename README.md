# Particle swarm optimization

A coursework implementation and a later experiment with a different velocity equation. Both optimize bounded continuous functions using personal, neighborhood and swarm-wide best positions.

[Coursework notebook](FuzzySystems-EvolutionaryComputing-Project.ipynb) · [Experimental implementation](pso_implementation.py) · [Assignment context](docs/assignment-context.md)

## Two versions

The **2024/25 submission** follows the assignment's explicit `candidate - best` equation (page 3) and required `softpy` interfaces. Its saved sphere example reports fitness −1.508873 after 10,000 iterations.

The **November 2026 experiment** uses the usual attractive direction, `best - position`, in a standalone NumPy implementation. It also changes initialization, boundary handling and final-population evaluation. This is a different algorithm and interface, not a replacement for the submission. The original notebook is unchanged.

## Run the experiment

```bash
git clone https://github.com/pdmdp/ParticleSwarmOptimizationProject.git
cd ParticleSwarmOptimizationProject
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python benchmark.py
```

The interface maximizes fitness: negate a loss to minimize it. [Usage and update rule](docs/variant.md) give a complete example. Original notebook dependencies are in `requirements-original.txt`.

## Ten-seed comparison

![Median best loss and interquartile range for the experimental variant](assets/convergence.png)

Each run uses 15 particles, three sampled neighbors, 300 updates and bounds `[−5.12, 5.12]` in both dimensions. Seeds are 0–9. The plot clips losses below `1e-16` for display; raw values are retained in [the benchmark results](results/benchmark.json).

| Function | Median final loss | Best | Worst |
| :--- | ---: | ---: | ---: |
| Sphere, 2D | 1.86e−43 | 3.97e−44 | 8.79e−43 |
| Rosenbrock, 2D | 0.00747 | 0.00000 | 0.62592 |
| Rastrigin, 2D | 0.99496 | 0.00000 | 0.99496 |

Sphere converges reliably in these runs; Rosenbrock and Rastrigin show seed dependence. These small benchmarks describe this configuration, without establishing superiority over other optimizers.

Six tests cover attraction, convergence, bounds, final-step evaluation, repeated fits, single-particle behavior and invalid inputs. They pass with Python 3.14.4 and NumPy 2.4.3.

**Matteo Pietro Di Pilato** · Fuzzy Systems and Evolutionary Computing, BSc in Artificial Intelligence. Code: [MIT](LICENSE).
