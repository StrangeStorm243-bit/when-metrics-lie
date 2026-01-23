import json
from pathlib import Path

from metrics_lie.spec import load_experiment_spec
from metrics_lie.schema import ResultBundle


def main() -> None:
    # Resolve path relative to project root (where pyproject.toml is)
    project_root = Path(__file__).parent.parent.parent
    spec_path = project_root / "examples" / "experiment_minimal.json"
    spec = load_experiment_spec(json.loads(spec_path.read_text()))

    # Pretend we ran something, produce an empty-but-valid bundle
    bundle = ResultBundle(
        run_id="RUN_DEMO",
        experiment_name=spec.name,
        metric_name=spec.metric,
    )

    print("[OK] Spec validated")
    print("[OK] ResultBundle created")
    print(bundle.to_pretty_json())


if __name__ == "__main__":
    main()
