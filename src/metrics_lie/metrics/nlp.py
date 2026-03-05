from __future__ import annotations

from typing import Sequence


def metric_rouge_l(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """ROUGE-L score via HuggingFace evaluate."""
    import evaluate

    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=list(y_pred), references=list(y_true))
    return float(result["rougeL"])


def metric_bleu(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """BLEU score via HuggingFace evaluate."""
    import evaluate

    bleu = evaluate.load("bleu")
    result = bleu.compute(
        predictions=list(y_pred),
        references=[[ref] for ref in y_true],
    )
    return float(result["bleu"])
