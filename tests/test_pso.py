import unittest
import numpy as np
from pso_implementation import ParticleCandidate, ParticleSwarmOptimizer


def sphere(p):
    return -float(np.dot(p.candidate, p.candidate))


class TestPSO(unittest.TestCase):
    def test_attraction_points_toward_best(self):
        p = ParticleCandidate(np.array([2.0]), np.array([0.0]), np.array([-5.]), np.array([5.]))
        target = p.copy(); target.candidate[:] = 0
        p.recombine(target, target, target, np.random.default_rng(7))
        p.mutate()
        self.assertTrue(0 < p.candidate[0] < 2)

    def test_sphere_convergence(self):
        pso = ParticleSwarmOptimizer(sphere, seed=42)
        result = pso.fit(300)
        self.assertLess(np.linalg.norm(result.candidate), 1e-4)
        self.assertTrue(np.all(np.diff(pso.history_) >= 0))

    def test_repeat_fit_resets_state_and_seed(self):
        pso = ParticleSwarmOptimizer(sphere, seed=4)
        a = pso.fit(30); first = list(pso.history_)
        b = pso.fit(30)
        np.testing.assert_array_equal(a.candidate, b.candidate)
        self.assertEqual(first, pso.history_)
        c = pso.fit(0)
        fresh = ParticleSwarmOptimizer(sphere, seed=4).fit(0)
        np.testing.assert_array_equal(c.candidate, fresh.candidate)

    def test_bounds_and_final_evaluation(self):
        evaluated = []
        def objective(p):
            self.assertTrue(np.all(p.candidate >= [-2, 3]))
            self.assertTrue(np.all(p.candidate <= [-1, 4]))
            evaluated.append(p.candidate.copy())
            return float(p.candidate.sum())
        pso = ParticleSwarmOptimizer(objective, pop_size=5, lower=[-2, 3], upper=[-1, 4], seed=6)
        result = pso.fit(7)
        self.assertEqual(len(evaluated), 5 * 8)
        self.assertEqual(pso.global_fitness_best, max(x.sum() for x in evaluated))
        np.testing.assert_array_equal(result.candidate, pso.global_best.candidate)
        result.candidate[:] = 100
        self.assertLess(pso.global_best.candidate.max(), 100)

    def test_single_particle_and_no_neighbors(self):
        for count in [0, 3]:
            pso = ParticleSwarmOptimizer(sphere, pop_size=1, n_neighbors=count, seed=8)
            self.assertTrue(np.isfinite(pso.fit(5).candidate).all())

    def test_invalid_parameters_and_objectives(self):
        for kwargs in [dict(pop_size=0), dict(n_neighbors=-1), dict(size=0),
                       dict(lower=3, upper=2), dict(lower=[0, 1, 2]), dict(upper=np.inf)]:
            with self.assertRaises(ValueError): ParticleSwarmOptimizer(sphere, **kwargs)
        with self.assertRaises(ValueError): ParticleSwarmOptimizer(sphere).fit(-1)
        with self.assertRaises(ValueError): ParticleSwarmOptimizer(lambda p: np.nan).fit(0)


if __name__ == '__main__':
    unittest.main()
