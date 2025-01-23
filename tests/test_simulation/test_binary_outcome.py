import numpy as np
import unittest

from ehr2vec.simulation.binary_outcome import (
    tbehrt,
    linear_logistic,
    simulate_outcome_from_embeddings,
)


class TestBinaryOutcome(unittest.TestCase):
    def setUp(self):
        # Replace pytest fixtures with setUp
        self.rng = np.random.RandomState(42)
        n_samples = 1000

        # Setup sample_data
        self.propensity_score = self.rng.normal(0, 1, size=n_samples)
        self.exposure = self.rng.binomial(1, 0.5, size=n_samples)

        # Setup embedding_data
        n_features_t = 50
        n_features_o = 30
        self.z_t = self.rng.normal(0, 1, size=(n_samples, n_features_t))
        self.z_o = self.rng.normal(0, 1, size=(n_samples, n_features_o))

    def test_shape_and_probability_range(self):
        # For tbehrt and linear_logistic which share the same signature
        for func in [tbehrt, linear_logistic]:
            with self.subTest(func=func.__name__):
                binary_outcome, probability = func(
                    self.propensity_score, self.exposure, a=0.5, b=1.0, c=0.3
                )
                self.assertEqual(binary_outcome.shape, self.exposure.shape)
                self.assertTrue(np.all(probability >= 0) and np.all(probability <= 1))

        # For simulate_outcome_from_embeddings which uses different args
        with self.subTest(func="simulate_outcome_from_embeddings"):
            binary_outcome, probability = simulate_outcome_from_embeddings(
                self.z_t, self.z_o, self.exposure, a=0.5, b=1.0, c=0.8, d=0.2
            )
            self.assertEqual(binary_outcome.shape, self.exposure.shape)
            self.assertTrue(np.all(probability >= 0) and np.all(probability <= 1))

    def test_exposure_effect(self):
        # If a >> 0, we expect more 1s when exposure=1
        exposure = np.array([0, 1] * 500)
        bin_outcome_low, _ = tbehrt(
            self.propensity_score, exposure, a=0.1, b=1.0, c=0.3
        )
        bin_outcome_high, _ = tbehrt(
            self.propensity_score, exposure, a=10.0, b=1.0, c=0.3
        )

        mean_low = np.mean(bin_outcome_low[exposure == 1])
        mean_high = np.mean(bin_outcome_high[exposure == 1])
        self.assertGreater(mean_high, mean_low)

    def test_simulate_outcome_from_embeddings_shape_mismatch(self):
        # z_t and z_o have inconsistent number of samples
        z_t = np.random.normal(0, 1, size=(100, 10))
        z_o = np.random.normal(0, 1, size=(99, 8))  # mismatch
        exposure = np.random.binomial(1, 0.5, size=100)

        with self.assertRaises(ValueError):
            simulate_outcome_from_embeddings(z_t, z_o, exposure, 0.5, 1.0, 0.8, 0.2)

    def test_simulate_outcome_from_embeddings_sparsity(self):
        t_sparsity = 0.7
        o_sparsity = 0.8
        n_runs = 10
        t_zero_proportions = []
        o_zero_proportions = []

        for _ in range(n_runs):
            _, _, W_t, W_o = simulate_outcome_from_embeddings(
                self.z_t,
                self.z_o,
                self.exposure,
                a=0.5,
                b=1.0,
                c=0.8,
                d=0.2,
                t_sparsity=t_sparsity,
                o_sparsity=o_sparsity,
                return_coefficients=True,
            )
            t_zero_proportions.append(np.mean(W_t == 0))
            o_zero_proportions.append(np.mean(W_o == 0))

        # Check if the actual average zero proportion is within some tolerance
        self.assertLess(abs(np.mean(t_zero_proportions) - t_sparsity), 0.1)
        self.assertLess(abs(np.mean(o_zero_proportions) - o_sparsity), 0.1)

    def test_reproducibility(self):
        z_t = np.random.normal(0, 1, size=(100, 10))
        z_o = np.random.normal(0, 1, size=(100, 8))
        exposure = np.random.binomial(1, 0.5, size=100)

        _, prob1 = simulate_outcome_from_embeddings(
            z_t, z_o, exposure, a=0.5, b=1.0, c=0.8, d=0.2
        )
        _, prob2 = simulate_outcome_from_embeddings(
            z_t, z_o, exposure, a=0.5, b=1.0, c=0.8, d=0.2
        )

        np.testing.assert_array_equal(prob1, prob2)


if __name__ == "__main__":
    unittest.main()
