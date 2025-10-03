![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-PSO-red.svg)
# Particle Swarm Optimization (PSO) Implementation

A Python implementation of the Particle Swarm Optimization algorithm for the Fuzzy Systems and Evolutionary Computing course (a.y. 2024-2025).

**Author:** Matteo Pietro Di Pilato  
**ID:** 525298

## Overview

This project implements Particle Swarm Optimization, a population-based meta-heuristic algorithm designed to efficiently explore and exploit high-dimensional search spaces. PSO operates by maintaining a swarm of particles, each encoding a candidate solution, with states defined through positional vectors and associated velocities.

## Algorithm Description

PSO updates particle velocities and positions at each iteration using the canonical update rule:

```
velocity ← (inertia) · velocity
         + r_l · w_l · (candidate - local_best)
         + r_n · w_n · (candidate - neighborhood_best)
         + r_g · w_g · (candidate - global_best)
```

Where:
- **inertia** (0.7): Weight for previous velocity
- **w_l** (0.1): Weight for attraction to personal best
- **w_n** (0.1): Weight for attraction to neighborhood best  
- **w_g** (0.8): Weight for attraction to global best
- **r_l, r_n, r_g**: Random coefficients ∈ [0,1]

## Features

- **ParticleCandidate Class**: Represents individual particles with position and velocity
- **ParticleSwarmOptimizer Class**: Manages the swarm and optimization loop
- **Neighborhood Dynamics**: Dynamic neighbor sampling for information sharing
- **Boundary Handling**: Particles remain within defined search space bounds
- **softpy Integration**: Full compatibility with the softpy library interface

## Installation

```bash
pip install softpy numpy scipy
```

## Usage

```python
import numpy as np
from pso_implementation import ParticleSwarmOptimizer, ParticleCandidate

# Define fitness function (to maximize)
def fitness_function(particle):
    return -np.sum(particle.candidate ** 2)

# Create PSO optimizer
pso = ParticleSwarmOptimizer(
    fitness_func=fitness_function,
    pop_size=15,
    n_neighbors=3,
    size=2,
    lower=np.array([-10.0, -10.0]),
    upper=np.array([10.0, 10.0])
)

# Run optimization
result = pso.fit(n_iters=10000)

print(f"Best solution: {result.candidate}")
print(f"Best fitness: {pso.global_fitness_best}")
```

## Project Structure

```
.
├── FuzzySystems-EvolutionaryComputing-Project.ipynb
├── README.md
├── requirements.txt
└── src/
    └── pso_implementation.py
```

## Implementation Details

### ParticleCandidate Class

Inherits from `FloatVectorCandidate` with the following key methods:

- `generate(size, lower, upper)`: Factory method for random particle initialization
- `mutate()`: Updates position using current velocity
- `recombine(local_best, neighborhood_best, global_best)`: Updates velocity using PSO formula

### ParticleSwarmOptimizer Class

Inherits from `MetaHeuristicsAlgorithm` with core functionality:

- `__init__()`: Initialize swarm parameters
- `fit(n_iters)`: Main optimization loop with fitness evaluation and particle updates

## Example Results

Testing on a simple quadratic function with optimum at [0, 0]:

```
Best solution found: [-0.789, 0.941]
Best fitness: -1.508873
Distance from optimum: 1.2284
```

## Requirements

- Python 3.7+
- numpy >= 1.24.0
- scipy >= 1.11.0
- softpy >= 0.1.2

## Course Information

This project was developed for the Fuzzy Systems and Evolutionary Computing course at the University, demonstrating practical implementation of swarm intelligence algorithms within a structured optimization framework.

## License

This project is for educational purposes as part of academic coursework.

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization
- softpy library documentation: https://github.com/evolutionary-computing/softpy
