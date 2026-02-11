import metrics_lie.scenarios.class_imbalance  # noqa: F401
import metrics_lie.scenarios.label_noise  # noqa: F401
import metrics_lie.scenarios.score_noise  # noqa: F401
import metrics_lie.scenarios.threshold_gaming  # noqa: F401
from metrics_lie.scenarios.registry import list_scenarios


def test_scenarios_registered():
    ids = list_scenarios()
    assert "label_noise" in ids
    assert "score_noise" in ids
    assert "class_imbalance" in ids
    assert "threshold_gaming" in ids
