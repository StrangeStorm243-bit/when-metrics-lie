"""Pre-built scenario suites for common evaluation patterns."""
from __future__ import annotations

standard_stress_suite: list[dict] = [
    {"id": "label_noise", "params": {"p": 0.05}},
    {"id": "label_noise", "params": {"p": 0.10}},
    {"id": "label_noise", "params": {"p": 0.20}},
    {"id": "score_noise", "params": {"sigma": 0.02}},
    {"id": "score_noise", "params": {"sigma": 0.05}},
    {"id": "score_noise", "params": {"sigma": 0.10}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.1}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.3}},
]

light_stress_suite: list[dict] = [
    {"id": "label_noise", "params": {"p": 0.10}},
    {"id": "score_noise", "params": {"sigma": 0.05}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.2}},
]

classification_suite: list[dict] = [
    {"id": "label_noise", "params": {"p": 0.05}},
    {"id": "label_noise", "params": {"p": 0.10}},
    {"id": "label_noise", "params": {"p": 0.20}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.1}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.3}},
    {"id": "threshold_gaming", "params": {}},
]

regression_suite: list[dict] = [
    {"id": "label_noise", "params": {"p": 0.05}},
    {"id": "label_noise", "params": {"p": 0.10}},
    {"id": "score_noise", "params": {"sigma": 0.02}},
    {"id": "score_noise", "params": {"sigma": 0.05}},
    {"id": "score_noise", "params": {"sigma": 0.10}},
]
