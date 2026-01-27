from metrics_lie.scenarios import class_imbalance, label_noise, score_noise, threshold_gaming  # noqa: F401
from metrics_lie.scenarios.registry import list_scenarios

def test_scenarios_registered():
    ids = list_scenarios()
    assert "label_noise" in ids
    assert "score_noise" in ids
    assert "class_imbalance" in ids
    assert "threshold_gaming" in ids
