# Plan: Phase 6 Stream B — KServe V2 Protocol Support

> **Execution model:** This plan was written by Opus for execution by Sonnet.
> Run: `claude --model claude-sonnet-4-6` then say "Execute plan .claude/plans/phase6-stream-b-kserve.md"

## Goal

Add KServe V2 Inference Protocol support to the existing HTTP adapter. When `protocol: "kserve_v2"` is set (or auto-detected from URL), the adapter formats requests/responses per the KServe V2 tensor spec.

## Context

- Existing HTTP adapter: `src/metrics_lie/model/adapters/http_adapter.py` (125 lines)
- Current protocol: `POST {"instances": X.tolist()}` → `{"predictions": [{"label": ..., "probability": ...}]}`
- KServe V2 protocol: `POST /v2/models/{name}/infer` with tensor format
- KServe V2 request: `{"inputs": [{"name": "input", "shape": [n, features], "datatype": "FP64", "data": [...]}]}`
- KServe V2 response: `{"outputs": [{"name": "output", "shape": [...], "data": [...]}]}`
- Existing test: `tests/test_v1_http_adapter.py` — 3 tests using `unittest.mock.patch("requests.post")`

## Prerequisites

- [ ] Read `src/metrics_lie/model/adapters/http_adapter.py` before modifying

## Tasks

### Task B1: Add KServe V2 protocol to HTTP adapter

**Files:**
- Modify: `src/metrics_lie/model/adapters/http_adapter.py`

**Steps:**

1. Update `__init__` to accept `protocol` and `model_name` parameters. Replace the entire `__init__` method:

```python
    def __init__(
        self,
        *,
        endpoint: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
        headers: dict[str, str] | None = None,
        protocol: str = "custom",
        model_name: str = "",
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._headers = headers or {"Content-Type": "application/json"}
        self._model_name = model_name

        # Auto-detect KServe V2 from URL
        if protocol == "custom" and "/v2/" in endpoint:
            self._protocol = "kserve_v2"
        else:
            self._protocol = protocol
```

2. Add KServe V2 request/response helpers. Add these methods after `__init__` and before `_call_endpoint`:

```python
    def _build_kserve_url(self) -> str:
        """Build KServe V2 inference URL."""
        if self._model_name:
            return f"{self._endpoint}/v2/models/{self._model_name}/infer"
        # If endpoint already has /v2/ path, use as-is
        return self._endpoint

    def _format_kserve_request(self, X: np.ndarray) -> dict[str, Any]:
        """Format input as KServe V2 tensor request."""
        return {
            "inputs": [
                {
                    "name": "input",
                    "shape": list(X.shape),
                    "datatype": "FP64",
                    "data": X.flatten().tolist(),
                }
            ]
        }

    def _parse_kserve_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse KServe V2 tensor response into prediction dicts.

        KServe V2 returns: {"outputs": [{"name": "output", "shape": [...], "data": [...]}]}
        We convert to our internal format: [{"label": ..., "probability": ...}, ...]
        """
        outputs = data.get("outputs", [])
        if not outputs:
            return []

        # Find the primary output
        primary = outputs[0]
        values = primary.get("data", [])
        shape = primary.get("shape", [])

        n_samples = shape[0] if shape else len(values)

        # If shape is [n, classes] — treat as probabilities
        if len(shape) == 2 and shape[1] > 1:
            n_classes = shape[1]
            preds = []
            for i in range(n_samples):
                row = values[i * n_classes : (i + 1) * n_classes]
                label = int(np.argmax(row))
                preds.append({"label": label, "probability": row})
            return preds

        # If shape is [n] — treat as labels or scores
        preds = []
        for i in range(n_samples):
            val = values[i] if i < len(values) else 0
            if isinstance(val, float) and 0.0 <= val <= 1.0:
                label = int(val >= self._threshold)
                preds.append({"label": label, "probability": [1.0 - val, val]})
            else:
                preds.append({"label": int(val)})
        return preds
```

3. Replace the existing `_call_endpoint` method with protocol-aware version:

```python
    def _call_endpoint(self, X: np.ndarray) -> list[dict[str, Any]]:
        import requests

        if self._protocol == "kserve_v2":
            url = self._build_kserve_url()
            payload = self._format_kserve_request(X)
        else:
            url = self._endpoint
            payload = {"instances": X.tolist()}

        resp = requests.post(
            url,
            json=payload,
            headers=self._headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        if self._protocol == "kserve_v2":
            return self._parse_kserve_response(data)
        return data.get("predictions", [])
```

4. Run: `cd /c/GitHubProjects/when-metrics-lie && python -c "from metrics_lie.model.adapters.http_adapter import HTTPAdapter; print('OK')"`
   Expected: OK

**Acceptance:** HTTPAdapter accepts `protocol="kserve_v2"` and `model_name` params. Auto-detects KServe from `/v2/` in URL. Default behavior (`protocol="custom"`) unchanged.

### Task B2: Verify existing tests still pass

**Steps:**

1. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m pytest tests/test_v1_http_adapter.py -v --tb=short`
   Expected: 3 tests pass (existing custom-protocol tests unchanged)

2. If any test fails, investigate. The existing tests use `endpoint="http://localhost:8080/predict"` which does NOT contain `/v2/`, so auto-detection should not trigger.

**Acceptance:** All 3 existing HTTP adapter tests pass unchanged.

### Task B3: Lint and full test suite

**Steps:**

1. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m ruff check src/metrics_lie/model/adapters/http_adapter.py --fix`
2. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5`
   Expected: All tests pass.

**Acceptance:** Lint clean, no regressions.

## Boundaries

**DO:**
- Follow steps exactly
- Only modify `http_adapter.py`
- Preserve backward compatibility (default protocol is "custom")

**DO NOT:**
- Create new adapter files (this is an enhancement to the existing adapter)
- Modify any other source files
- Create a git commit (parent agent handles this)

## Escalation Triggers

Stop and flag for Opus review if:
- Existing HTTP adapter tests fail after changes
- The `_call_endpoint` signature change breaks other callers
- Import errors

When escalating, write to `.claude/plans/phase6-stream-b-blockers.md`.

## Verification

After all tasks complete:
- [ ] `python -c "from metrics_lie.model.adapters.http_adapter import HTTPAdapter; a = HTTPAdapter(endpoint='http://x/v2/models/m/infer'); print(a._protocol)"` prints `kserve_v2`
- [ ] `python -c "from metrics_lie.model.adapters.http_adapter import HTTPAdapter; a = HTTPAdapter(endpoint='http://x/predict'); print(a._protocol)"` prints `custom`
- [ ] `python -m ruff check src/` passes
- [ ] `python -m pytest tests/ -x -q` passes
