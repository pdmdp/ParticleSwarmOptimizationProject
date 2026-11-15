# Experimental NumPy variant


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


## Update rule

With independent scalar random coefficients drawn uniformly from `[0, 1)`, the experimental variant uses:

```text
v = 0.7 v
  + 0.1 r_personal (personal_best - position)
  + 0.1 r_neighbor (neighborhood_best - position)
  + 0.8 r_global   (global_best - position)
```

Positions are clipped to their bounds and outward velocity is set to zero on clipped dimensions. The weights retain the coursework configuration; they are not a tuned recommendation for arbitrary objectives.
