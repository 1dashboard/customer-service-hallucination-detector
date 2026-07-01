"""Evaluation engine: compare detection results against ground truth."""

from __future__ import annotations

import json
from pathlib import Path

from .models import EvaluationMetrics


def compute_metrics(
    ground_truth: list[dict],
    results: list[dict],
) -> EvaluationMetrics:
    """Compute accuracy, precision, recall, F1 from detection results vs ground truth.

    Args:
        ground_truth: List of {id, is_hallucination, hallucination_type, detail}
        results: List of {id, is_hallucination, ...}

    Returns:
        EvaluationMetrics with all computed values and misclassified cases.
    """
    gt_map = {item["id"]: item for item in ground_truth}

    tp = tn = fp = fn = 0
    fp_cases: list[dict] = []
    fn_cases: list[dict] = []

    for result in results:
        rid = result["id"]
        gt = gt_map.get(rid)

        if gt is None:
            continue

        pred = result.get("is_hallucination")
        actual = gt["is_hallucination"]

        if pred is True and actual is True:
            tp += 1
        elif pred is False and actual is False:
            tn += 1
        elif pred is True and actual is False:
            fp += 1
            fp_cases.append({
                "id": rid,
                "predicted": "hallucination",
                "actual": "non-hallucination",
                "ground_truth_detail": gt.get("detail", ""),
                "detection_reason": result.get("reason", ""),
                "output_type": result.get("output_type", ""),
            })
        elif pred is False and actual is True:
            fn += 1
            fn_cases.append({
                "id": rid,
                "predicted": "non-hallucination",
                "actual": gt.get("hallucination_type", "unknown"),
                "ground_truth_detail": gt.get("detail", ""),
                "detection_reason": result.get("reason", ""),
            })
        # pred is None (uncertain) → treat as neither TP nor FP

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvaluationMetrics(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        false_positive_cases=fp_cases,
        false_negative_cases=fn_cases,
    )


def evaluate_from_files(
    results_file: str,
    ground_truth_file: str,
) -> EvaluationMetrics:
    """Load results and ground truth from JSON files and compute metrics."""
    with open(results_file, encoding="utf-8") as f:
        results = json.load(f)
    with open(ground_truth_file, encoding="utf-8") as f:
        ground_truth = json.load(f)
    return compute_metrics(ground_truth, results)


def print_evaluation_report(metrics: EvaluationMetrics) -> None:
    """Print a formatted evaluation report."""
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Accuracy:  {metrics.accuracy:.2%}")
    print(f"  Precision: {metrics.precision:.2%}")
    print(f"  Recall:    {metrics.recall:.2%}")
    print(f"  F1 Score:  {metrics.f1:.2%}")
    print("-" * 60)
    print(f"  True Positives:  {metrics.true_positives}")
    print(f"  True Negatives:  {metrics.true_negatives}")
    print(f"  False Positives: {metrics.false_positives}")
    print(f"  False Negatives: {metrics.false_negatives}")
    print("=" * 60)

    if metrics.false_negatives > 0:
        print(f"\n  MISSED HALLUCINATIONS ({len(metrics.false_negative_cases)}):")
        for case in metrics.false_negative_cases:
            print(f"    [{case['id']}] Actual: {case['actual']}")
            print(f"           GT: {case['ground_truth_detail'][:120]}")
            print(f"           Detection: {case['detection_reason'][:120]}")
            print()

    if metrics.false_positives > 0:
        print(f"\n  FALSE ALARMS ({len(metrics.false_positive_cases)}):")
        for case in metrics.false_positive_cases:
            print(f"    [{case['id']}] Type: {case['output_type']}")
            print(f"           GT: {case['ground_truth_detail'][:120]}")
            print(f"           Detection: {case['detection_reason'][:120]}")
            print()
