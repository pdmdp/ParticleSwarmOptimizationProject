"""Run three small, seeded optimization checks and plot their convergence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pso_implementation import ParticleSwarmOptimizer


def run():
    objectives = {
        'Sphere (2D)': lambda x: float(np.sum(x*x)),
        'Rosenbrock (2D)': lambda x: float(100*(x[1]-x[0]**2)**2+(1-x[0])**2),
        'Rastrigin (2D)': lambda x: float(20+np.sum(x*x-10*np.cos(2*np.pi*x))),
    }
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout='constrained')
    for ax, (name, loss) in zip(axes, objectives.items()):
        runs = []
        for seed in range(10):
            model = ParticleSwarmOptimizer(lambda p: -loss(p.candidate),
                                          lower=-5.12, upper=5.12, seed=seed)
            best = model.fit(300)
            runs.append({'seed': seed, 'loss': -model.global_fitness_best,
                         'position': best.candidate.tolist(),
                         'history': [-x for x in model.history_]})
        traces = np.array([r['history'] for r in runs])
        results[name] = {'median_final_loss': float(np.median(traces[:, -1])),
                         'min_final_loss': float(traces[:, -1].min()),
                         'max_final_loss': float(traces[:, -1].max()), 'runs': runs}
        median = np.median(traces, axis=0)
        lo, hi = np.quantile(traces, [.25, .75], axis=0)
        ax.plot(np.maximum(median, 1e-16), color='#317c9c')
        ax.fill_between(np.arange(len(median)), np.maximum(lo, 1e-16),
                        np.maximum(hi, 1e-16), alpha=.2, color='#317c9c')
        ax.set(yscale='log', xlabel='Position updates', title=name)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Best loss so far')
    root = Path(__file__).resolve().parent
    (root/'results').mkdir(exist_ok=True); (root/'assets').mkdir(exist_ok=True)
    (root/'results/benchmark.json').write_text(json.dumps({
        'provenance': 'Experimental attractive NumPy variant, benchmark run on 2026-11-15. Its update rule differs from the course assignment; these are not submission results.',
        'settings': {'particles': 15, 'neighbors': 3, 'iterations': 300,
                     'bounds': [-5.12, 5.12], 'seeds': list(range(10))},
        'results': results,
    }, indent=2)+'\n')
    fig.savefig(root/'assets/convergence.png', dpi=180, facecolor='white')
    for name, data in results.items():
        print(name, {k:v for k,v in data.items() if k != 'runs'})


if __name__ == '__main__':
    run()
