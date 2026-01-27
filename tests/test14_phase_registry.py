from metrics_lie.scenarios import label_noise, score_noise  # noqa: F401
from metrics_lie.scenarios.registry import list_scenarios

def test_scenarios_registered():
    ids = list_scenarios()
    assert "label_noise" in ids
    assert "score_noise" in ids
