import json
from pathlib import Path

from metrics_lie.spec import load_experiment_spec


def test_example_spec_parses():
    spec = load_experiment_spec(json.loads(Path("examples/experiment_minimal.json").read_text()))
    assert spec.name
    assert spec.metric in {"auc", "logloss", "accuracy"}
