# Assignment context

The reference is **Project Assignment – Fuzzy Systems and Evolutionary Computing (a.y. 2024/2025)**, file `Project_Fuzzy_2024_2025.pdf`, pages 2–4.

Page 3 explicitly defines the velocity update as:

```text
velocity = inertia * velocity
         + rl * wl * (candidate - local_best)
         + rn * wn * (candidate - neighborhood_best)
         + rg * wg * (candidate - best)
```

The submitted notebook implements that direction. A difference from the commonly used attractive PSO equation does not show that the submission violates the assignment. The notebook is retained without changes.

The November 2026 NumPy experiment uses `best - candidate` for the three attraction terms. It intentionally changes the specified dynamics and is evaluated separately. It also differs in interface and iteration semantics:

| Aspect | Course specification / submitted notebook | Later experiment |
| :--- | :--- | :--- |
| Candidate base class | `softpy.FloatVectorCandidate` | Standalone dataclass |
| Optimizer base class | `softpy.MetaHeuristicsAlgorithm` | Standalone Python class |
| Direction in velocity terms | `candidate - best` | `best - candidate` |
| Iteration schedule | Evaluate, update bests, then move | Evaluate initially and after each move |
| Initialization | NumPy global random state | Local seeded generator |
| Bounds in submitted code | Clip position | Clip position and stop outward velocity |

The experiment's convergence curves should not be presented as the original assignment results or as evidence that the specified equation was implemented incorrectly. An earlier repository description called the change a correction; that characterization was removed after checking the assignment text.
