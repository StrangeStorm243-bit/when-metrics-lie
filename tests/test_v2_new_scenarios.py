"""Tests for v2 new stress-test scenarios."""
from __future__ import annotations

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import create_scenario, list_scenarios


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _binary_ctx() -> ScenarioContext:
    return ScenarioContext(task="binary_classification", surface_type="probability")


def _multiclass_ctx() -> ScenarioContext:
    return ScenarioContext(task="multiclass_classification", surface_type="probability", n_classes=3)


def _regression_ctx() -> ScenarioContext:
    return ScenarioContext(task="regression", surface_type="continuous")


def _text_ctx() -> ScenarioContext:
    return ScenarioContext(task="text_classification", surface_type="label")


# ---------------------------------------------------------------------------
# Scenario 1: missing_features
# ---------------------------------------------------------------------------

class TestMissingFeatures:
    def test_apply_2d(self):
        from metrics_lie.scenarios.missing_features import MissingFeaturesScenario
        s = MissingFeaturesScenario(drop_rate=0.5)
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        assert yt.shape == y_true.shape
        assert ys.shape == y_score.shape
        assert np.any(np.isnan(ys))  # some values dropped

    def test_apply_1d_passthrough(self):
        from metrics_lie.scenarios.missing_features import MissingFeaturesScenario
        s = MissingFeaturesScenario()
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        np.testing.assert_array_equal(ys, y_score)

    def test_describe(self):
        from metrics_lie.scenarios.missing_features import MissingFeaturesScenario
        s = MissingFeaturesScenario()
        d = s.describe()
        assert d["id"] == "missing_features"
        assert "drop_rate" in d

    def test_registry(self):
        s = create_scenario("missing_features", {"drop_rate": 0.3})
        assert s.id == "missing_features"


# ---------------------------------------------------------------------------
# Scenario 2: feature_corruption
# ---------------------------------------------------------------------------

class TestFeatureCorruption:
    def test_apply_2d(self):
        from metrics_lie.scenarios.feature_corruption import FeatureCorruptionScenario
        s = FeatureCorruptionScenario(corruption_rate=0.5, noise_scale=3.0)
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        assert ys.shape == y_score.shape
        assert not np.array_equal(ys, y_score)  # noise added

    def test_apply_1d_passthrough(self):
        from metrics_lie.scenarios.feature_corruption import FeatureCorruptionScenario
        s = FeatureCorruptionScenario()
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        np.testing.assert_array_equal(ys, y_score)

    def test_describe(self):
        from metrics_lie.scenarios.feature_corruption import FeatureCorruptionScenario
        s = FeatureCorruptionScenario()
        d = s.describe()
        assert d["id"] == "feature_corruption"

    def test_registry(self):
        s = create_scenario("feature_corruption", {})
        assert s.id == "feature_corruption"


# ---------------------------------------------------------------------------
# Scenario 3: covariate_shift
# ---------------------------------------------------------------------------

class TestCovariateShift:
    def test_apply_2d(self):
        from metrics_lie.scenarios.covariate_shift import CovariateShiftScenario
        s = CovariateShiftScenario(shift_scale=2.0)
        y_true = np.array([0, 1, 0])
        y_score = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        assert ys.shape == y_score.shape
        # All rows shifted by same per-column offset
        diffs = ys - y_score
        np.testing.assert_array_almost_equal(diffs[0], diffs[1])

    def test_apply_1d_passthrough(self):
        from metrics_lie.scenarios.covariate_shift import CovariateShiftScenario
        s = CovariateShiftScenario()
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        np.testing.assert_array_equal(ys, y_score)

    def test_describe(self):
        from metrics_lie.scenarios.covariate_shift import CovariateShiftScenario
        s = CovariateShiftScenario()
        assert s.describe()["id"] == "covariate_shift"

    def test_registry(self):
        s = create_scenario("covariate_shift", {"shift_scale": 1.5})
        assert s.id == "covariate_shift"


# ---------------------------------------------------------------------------
# Scenario 4: typo_injection
# ---------------------------------------------------------------------------

class TestTypoInjection:
    def test_apply_strings(self):
        from metrics_lie.scenarios.typo_injection import TypoInjectionScenario
        s = TypoInjectionScenario(typo_rate=0.5)
        y_true = np.array([0, 1])
        y_score = np.array(["hello world", "test string"], dtype=object)
        yt, ys = s.apply(y_true, y_score, _rng(), _text_ctx())
        assert ys.shape == y_score.shape
        # With 50% typo rate, at least one string should differ
        assert any(ys[i] != y_score[i] for i in range(len(ys)))

    def test_apply_numeric_passthrough(self):
        from metrics_lie.scenarios.typo_injection import TypoInjectionScenario
        s = TypoInjectionScenario()
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        np.testing.assert_array_equal(ys, y_score)

    def test_describe(self):
        from metrics_lie.scenarios.typo_injection import TypoInjectionScenario
        assert TypoInjectionScenario().describe()["id"] == "typo_injection"

    def test_registry(self):
        s = create_scenario("typo_injection", {})
        assert s.id == "typo_injection"


# ---------------------------------------------------------------------------
# Scenario 5: synonym_replacement
# ---------------------------------------------------------------------------

class TestSynonymReplacement:
    def test_apply_with_synonyms(self):
        from metrics_lie.scenarios.synonym_replacement import SynonymReplacementScenario
        s = SynonymReplacementScenario(replace_rate=1.0)
        y_true = np.array([0, 1])
        y_score = np.array(["good bad", "fast slow"], dtype=object)
        yt, ys = s.apply(y_true, y_score, _rng(), _text_ctx())
        # All synonym words should be replaced at rate=1.0
        assert "good" not in ys[0]
        assert "bad" not in ys[0]

    def test_apply_numeric_passthrough(self):
        from metrics_lie.scenarios.synonym_replacement import SynonymReplacementScenario
        s = SynonymReplacementScenario()
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        np.testing.assert_array_equal(ys, y_score)

    def test_describe(self):
        from metrics_lie.scenarios.synonym_replacement import SynonymReplacementScenario
        assert SynonymReplacementScenario().describe()["id"] == "synonym_replacement"

    def test_registry(self):
        s = create_scenario("synonym_replacement", {})
        assert s.id == "synonym_replacement"


# ---------------------------------------------------------------------------
# Scenario 6: demographic_swap
# ---------------------------------------------------------------------------

class TestDemographicSwap:
    def test_apply_swaps(self):
        from metrics_lie.scenarios.demographic_swap import DemographicSwapScenario
        s = DemographicSwapScenario()
        y_true = np.array([0, 1])
        y_score = np.array(["he is a boy", "she is a girl"], dtype=object)
        yt, ys = s.apply(y_true, y_score, _rng(), _text_ctx())
        assert "she" in ys[0]
        assert "girl" in ys[0]
        assert "he" in ys[1]
        assert "boy" in ys[1]

    def test_apply_numeric_passthrough(self):
        from metrics_lie.scenarios.demographic_swap import DemographicSwapScenario
        s = DemographicSwapScenario()
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        np.testing.assert_array_equal(ys, y_score)

    def test_describe(self):
        from metrics_lie.scenarios.demographic_swap import DemographicSwapScenario
        assert DemographicSwapScenario().describe()["id"] == "demographic_swap"

    def test_registry(self):
        s = create_scenario("demographic_swap", {})
        assert s.id == "demographic_swap"


# ---------------------------------------------------------------------------
# Scenario 7: temporal_shift
# ---------------------------------------------------------------------------

class TestTemporalShift:
    def test_apply_2d_rolls(self):
        from metrics_lie.scenarios.temporal_shift import TemporalShiftScenario
        s = TemporalShiftScenario(shift_fraction=0.5)
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        assert ys.shape == y_score.shape
        assert not np.array_equal(ys, y_score)

    def test_apply_1d_shifts(self):
        from metrics_lie.scenarios.temporal_shift import TemporalShiftScenario
        s = TemporalShiftScenario(shift_fraction=0.1)
        y_true = np.array([1.0, 2.0, 3.0])
        y_score = np.array([1.0, 2.0, 3.0])
        yt, ys = s.apply(y_true, y_score, _rng(), _regression_ctx())
        assert ys.shape == y_score.shape
        # 1D values shifted by fraction of std
        assert not np.array_equal(ys, y_score)

    def test_describe(self):
        from metrics_lie.scenarios.temporal_shift import TemporalShiftScenario
        assert TemporalShiftScenario().describe()["id"] == "temporal_shift"

    def test_registry(self):
        s = create_scenario("temporal_shift", {})
        assert s.id == "temporal_shift"


# ---------------------------------------------------------------------------
# Scenario 8: label_quality
# ---------------------------------------------------------------------------

class TestLabelQuality:
    def test_apply_binary(self):
        from metrics_lie.scenarios.label_quality import LabelQualityScenario
        s = LabelQualityScenario(error_rate=0.5)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        yt, ys = s.apply(y_true, y_score, _rng(), _binary_ctx())
        assert yt.shape == y_true.shape
        assert not np.array_equal(yt, y_true)  # labels flipped
        np.testing.assert_array_equal(ys, y_score)  # scores unchanged

    def test_apply_multiclass(self):
        from metrics_lie.scenarios.label_quality import LabelQualityScenario
        s = LabelQualityScenario(error_rate=0.5)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_score = np.array([
            [0.34, 0.33, 0.33], [0.33, 0.34, 0.33], [0.33, 0.33, 0.34],
            [0.34, 0.33, 0.33], [0.33, 0.34, 0.33], [0.33, 0.33, 0.34],
            [0.34, 0.33, 0.33], [0.33, 0.34, 0.33], [0.33, 0.33, 0.34],
            [0.34, 0.33, 0.33],
        ])
        ctx = _multiclass_ctx()
        yt, ys = s.apply(y_true, y_score, _rng(), ctx)
        assert not np.array_equal(yt, y_true)

    def test_apply_regression(self):
        from metrics_lie.scenarios.label_quality import LabelQualityScenario
        s = LabelQualityScenario(error_rate=0.3)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_score = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        ctx = _regression_ctx()
        yt, ys = s.apply(y_true, y_score, _rng(), ctx)
        assert not np.array_equal(yt, y_true)

    def test_describe(self):
        from metrics_lie.scenarios.label_quality import LabelQualityScenario
        assert LabelQualityScenario().describe()["id"] == "label_quality"

    def test_registry(self):
        s = create_scenario("label_quality", {})
        assert s.id == "label_quality"


# ---------------------------------------------------------------------------
# Registry & compatibility
# ---------------------------------------------------------------------------

class TestRegistryAndCompat:
    def test_all_new_scenarios_registered(self):
        names = list_scenarios()
        expected = [
            "missing_features", "feature_corruption", "covariate_shift",
            "typo_injection", "synonym_replacement", "demographic_swap",
            "temporal_shift", "label_quality",
        ]
        for name in expected:
            assert name in names, f"{name} not in registry"

    def test_task_compat_binary(self):
        from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT
        bc = SCENARIO_TASK_COMPAT["binary_classification"]
        for s in ["missing_features", "feature_corruption", "covariate_shift",
                   "demographic_swap", "temporal_shift", "label_quality"]:
            assert s in bc, f"{s} missing from binary_classification compat"

    def test_task_compat_multiclass(self):
        from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT
        mc = SCENARIO_TASK_COMPAT["multiclass_classification"]
        for s in ["missing_features", "feature_corruption", "covariate_shift",
                   "demographic_swap", "temporal_shift", "label_quality"]:
            assert s in mc, f"{s} missing from multiclass_classification compat"

    def test_task_compat_regression(self):
        from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT
        reg = SCENARIO_TASK_COMPAT["regression"]
        for s in ["missing_features", "feature_corruption", "covariate_shift",
                   "temporal_shift", "label_quality"]:
            assert s in reg, f"{s} missing from regression compat"

    def test_task_compat_text_classification(self):
        from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT
        tc = SCENARIO_TASK_COMPAT["text_classification"]
        for s in ["typo_injection", "synonym_replacement", "demographic_swap",
                   "label_quality", "label_noise"]:
            assert s in tc, f"{s} missing from text_classification compat"

    def test_task_compat_text_generation(self):
        from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT
        tg = SCENARIO_TASK_COMPAT["text_generation"]
        for s in ["typo_injection", "synonym_replacement", "demographic_swap"]:
            assert s in tg, f"{s} missing from text_generation compat"
