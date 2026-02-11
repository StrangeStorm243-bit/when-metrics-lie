import json
from pathlib import Path

from metrics_lie.experiments.datasets import dataset_fingerprint_csv
from metrics_lie.experiments.definition import ExperimentDefinition
from metrics_lie.spec import load_experiment_spec


def _load_spec(path: str):
    return load_experiment_spec(json.loads(Path(path).read_text()))


def test_same_spec_and_dataset_same_experiment_id(tmp_path):
    spec_path = "examples/experiment_minimal.json"
    ds_path = "data/demo_binary.csv"

    spec1 = _load_spec(spec_path)
    spec2 = _load_spec(spec_path)

    fp1 = dataset_fingerprint_csv(ds_path)
    fp2 = dataset_fingerprint_csv(ds_path)

    exp1 = ExperimentDefinition.from_spec(spec1, dataset_fingerprint=fp1)
    exp2 = ExperimentDefinition.from_spec(spec2, dataset_fingerprint=fp2)

    assert exp1.experiment_id == exp2.experiment_id


def test_changing_seed_changes_experiment_id(tmp_path, monkeypatch):
    spec_path = "examples/experiment_minimal.json"
    ds_path = "data/demo_binary.csv"

    base_spec = _load_spec(spec_path)

    spec_a = base_spec.model_copy()
    spec_b = base_spec.model_copy()
    spec_b.seed = base_spec.seed + 1

    fp = dataset_fingerprint_csv(ds_path)

    exp_a = ExperimentDefinition.from_spec(spec_a, dataset_fingerprint=fp)
    exp_b = ExperimentDefinition.from_spec(spec_b, dataset_fingerprint=fp)

    assert exp_a.experiment_id != exp_b.experiment_id
