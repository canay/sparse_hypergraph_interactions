from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

import analyze_results as analysis  # noqa: E402
import sc_shil_experiment as experiment  # noqa: E402


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = {
            "id": "T01",
            "kind": "mixed",
            "n": 400,
            "d": 16,
            "noise_sd": 0.65,
            "rho": 0.0,
            "signal_scale": 1.0,
            "threshold_quantile": 0.5,
            "heredity": "violated",
            "density": "sparse",
        }

    def test_generator_is_deterministic_and_balanced(self) -> None:
        left = experiment.make_synthetic(self.spec, 123)
        right = experiment.make_synthetic(self.spec, 123)
        np.testing.assert_array_equal(left.X, right.X)
        np.testing.assert_array_equal(left.y, right.y)
        self.assertEqual(left.X.shape, (400, 16))
        self.assertEqual(left.true_edges, {(0, 1), (2, 3), (4, 5, 6), (7, 8, 9)})
        self.assertEqual(np.bincount(left.y).tolist(), [200, 200])

    def test_redundant_scenario_has_declared_equivalence_groups(self) -> None:
        spec = {**self.spec, "id": "T11", "kind": "redundant_mixed", "d": 20, "proxy_correlation": 0.95}
        ds = experiment.make_synthetic(spec, 7)
        self.assertEqual(len(ds.equivalence_groups), 4)
        self.assertIn((16, 17), ds.equivalence_groups[0])


class SplitAndSelectionTests(unittest.TestCase):
    def test_outer_split_is_disjoint_and_train_scaled(self) -> None:
        spec = {
            "id": "T01", "kind": "pair", "n": 500, "d": 10,
            "noise_sd": 0.65, "rho": 0.0, "signal_scale": 1.0,
            "threshold_quantile": 0.5, "heredity": "violated", "density": "sparse"
        }
        ds = experiment.make_synthetic(spec, 91)
        split = experiment.outer_split_scale(ds, 91)
        self.assertEqual(len(split["train_idx"]), 300)
        self.assertEqual(len(split["val_idx"]), 100)
        self.assertEqual(len(split["test_idx"]), 100)
        np.testing.assert_allclose(split["X_train"].mean(axis=0), 0.0, atol=1e-12)

    def test_complementary_halves_are_stratified_disjoint_and_complete(self) -> None:
        y = np.array([0] * 40 + [1] * 20)
        halves = experiment.stratified_complementary_halves(y, 3, 5)
        self.assertEqual(len(halves), 6)
        for left, right in zip(halves[::2], halves[1::2]):
            self.assertFalse(set(left) & set(right))
            self.assertEqual(set(np.r_[left, right]), set(range(len(y))))
            self.assertEqual(np.bincount(y[left]).tolist(), [20, 10])
            self.assertEqual(np.bincount(y[right]).tolist(), [20, 10])

    def test_frequencies_from_rankings(self) -> None:
        rankings = [np.array([0, 1, 2]), np.array([0, 2, 1])]
        freq = experiment.frequencies_from_rankings(rankings, [1, 2], 3)
        np.testing.assert_allclose(freq[1], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(freq[2], [1.0, 0.5, 0.5])

    def test_one_se_rule_prefers_smallest_support(self) -> None:
        path = [
            {"q": 8, "pi": 0.6, "support_size": 5, "support": ((0, 1),) * 5, "validation_log_loss": 0.500, "validation_log_loss_se": 0.010},
            {"q": 4, "pi": 0.9, "support_size": 2, "support": ((0, 1), (2, 3)), "validation_log_loss": 0.507, "validation_log_loss_se": 0.012},
            {"q": 4, "pi": 0.8, "support_size": 3, "support": ((0, 1), (2, 3), (4, 5)), "validation_log_loss": 0.503, "validation_log_loss_se": 0.011},
        ]
        chosen = experiment.select_one_se(path)
        self.assertEqual(chosen["support_size"], 2)
        self.assertEqual(chosen["pi"], 0.9)


class MetricTests(unittest.TestCase):
    def test_interactions_are_clipped(self) -> None:
        X = np.array([[10.0, 10.0, 10.0], [-10.0, 10.0, 10.0]])
        Z = experiment.interaction_matrix(X, [(0, 1), (0, 1, 2)], 12.0)
        np.testing.assert_array_equal(Z, [[12.0, 12.0], [-12.0, -12.0]])

    def test_support_metrics(self) -> None:
        metrics = experiment.support_metrics([(0, 1), (2, 3), (8, 9)], {(0, 1), (2, 3), (4, 5)})
        self.assertAlmostEqual(metrics["support_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["support_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["support_f1"], 2 / 3)
        self.assertEqual(metrics["false_inclusions"], 1)

    def test_nogueira_is_one_for_identical_supports(self) -> None:
        supports = [{(0, 1), (2, 3)} for _ in range(5)]
        universe = [(0, 1), (0, 2), (1, 2), (2, 3)]
        self.assertAlmostEqual(analysis.nogueira_stability(supports, universe), 1.0)


class RankingSmokeTests(unittest.TestCase):
    def test_shil_ranking_is_deterministic(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(120, 6))
        y = (X[:, 0] * X[:, 1] + rng.normal(scale=0.3, size=120) > 0).astype(int)
        edges = experiment.candidate_edges(6, [2])
        Z = experiment.interaction_matrix(X, edges)
        fit_idx = np.arange(90)
        val_idx = np.arange(90, 120)
        config = {"epochs": 12, "learning_rate": 0.025, "gate_l1": 0.0008, "l2": 0.0002, "patience": 5}
        left = experiment.fit_shil_ranking(X, Z, y, fit_idx, val_idx, 17, config)
        right = experiment.fit_shil_ranking(X, Z, y, fit_idx, val_idx, 17, config)
        np.testing.assert_array_equal(left.ranked_indices, right.ranked_indices)
        self.assertEqual(left.ranked_indices[0], edges.index((0, 1)))


class ProvenanceKeyTests(unittest.TestCase):
    def test_real_data_config_uses_verified_official_uci_identity(self) -> None:
        config = experiment.json.loads((RUN_ROOT / "config" / "full_config.json").read_text(encoding="utf-8"))
        real = config["real_dataset"]
        self.assertEqual(real["uci_dataset_id"], 602)
        self.assertEqual(real["uci_doi"], "10.24432/C50S4B")
        self.assertEqual(real["expected_rows"], 13611)
        self.assertEqual(real["expected_features"], 16)
        self.assertEqual(real["expected_classes"], 7)
        self.assertEqual(len(real["expected_feature_names"]), 16)
        self.assertEqual(len(real["archive_sha256"]), 64)
        self.assertEqual(len(real["arff_sha256"]), 64)

    def test_run_dataset_preserves_outer_split_seed_in_every_output(self) -> None:
        config = experiment.json.loads((RUN_ROOT / "config" / "full_config.json").read_text(encoding="utf-8"))
        spec = {**config["scenarios"][0], "n": 320, "d": 10}
        dataset = experiment.make_synthetic(spec, 809)
        result = experiment.run_dataset(dataset, 809, config, n_pairs=1)
        for collection in result.values():
            self.assertTrue(collection)
            self.assertEqual({int(row["split_seed"]) for row in collection}, {809})
            self.assertEqual({int(row["data_seed"]) for row in collection}, {809})
        self.assertEqual(
            sum(int(row["warning_count"]) for row in result["base_fit_diagnostics"]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
