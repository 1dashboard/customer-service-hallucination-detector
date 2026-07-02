"""Detection pipeline orchestrator: Stage 1 (rules) → Stage 2 (LLM)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from .classifier import classify, classify_llm_result
from .db import get_connection, init_db
from .llm_judge import judge
from .models import Confidence, DetectionInput, DetectionLayer, DetectionResult, OutputType
from .rules import run_stage1


def detect_single(item: DetectionInput) -> DetectionResult:
    """Run the full detection pipeline on a single reply item."""
    reply_id = item.id or "unknown"
    user_question = item.user_question
    system_reply = item.system_reply
    knowledge_base = item.knowledge_base

    # Stage 1: Rule engine
    result = run_stage1(reply_id, user_question, system_reply, knowledge_base)

    # Classify L1/L2 results
    result = classify(result, user_question, knowledge_base)

    # If Stage 1 gave HIGH confidence, return directly
    if result.confidence == Confidence.HIGH and result.is_hallucination is True:
        return result

    # Stage 2: LLM semantic judgment for LOW-confidence cases
    llm_result = judge(reply_id, user_question, system_reply, knowledge_base)

    if llm_result["is_hallucination"] is True:
        result.is_hallucination = True
        result.detection_layer = DetectionLayer.L3_UNSUPPORTED_CLAIM
        result.confidence = Confidence(llm_result.get("confidence", "中"))
        result.reason = llm_result["reason"]
        result = classify_llm_result(result, llm_result.get("subtype", ""))
    elif llm_result["is_hallucination"] is False:
        result.is_hallucination = False
        result.confidence = Confidence(llm_result.get("confidence", "中"))
        result.reason = llm_result["reason"]
        result.detection_layer = None
        result.output_type = None
    else:
        # LLM call failed, keep Stage 1 result as-is
        pass

    return result


def detect_batch(items: list[dict]) -> list[DetectionResult]:
    """Run detection on a batch of reply items with concurrent LLM calls."""
    import time as _time
    _t0 = _time.time()

    # Phase 1: Stage 1 (rules) for all items — fast, no API calls
    stage1_results: list[DetectionResult] = []
    llm_queue: list[tuple[int, DetectionInput, DetectionResult]] = []  # (index, input, stage1_result)

    for i, item in enumerate(items):
        input_item = DetectionInput(
            id=item.get("id", str(i)),
            user_question=item["user_question"],
            system_reply=item["system_reply"],
            knowledge_base=item["knowledge_base"],
        )
        _t1 = _time.time()
        result = run_stage1(input_item.id, input_item.user_question, input_item.system_reply, input_item.knowledge_base)
        result = classify(result, input_item.user_question, input_item.knowledge_base)
        _dt = (_time.time() - _t1) * 1000

        if result.confidence == Confidence.HIGH and result.is_hallucination is True:
            print(f"  [{i+1}/{len(items)}] {input_item.id} → Stage1 HIGH ({result.output_type.value if result.output_type else '?'} | {_dt:.0f}ms)")
            stage1_results.append(result)
        else:
            print(f"  [{i+1}/{len(items)}] {input_item.id} → LLM queue ({_dt:.0f}ms)")
            stage1_results.append(result)
            llm_queue.append((i, input_item, result))

    _t_s1 = _time.time() - _t0
    print(f"  Stage1 done: {len(items) - len(llm_queue)} caught by rules, {len(llm_queue)} need LLM ({_t_s1:.1f}s)")

    # Phase 2: Concurrent LLM judgment for items that need it
    if llm_queue:
        _t_llm_start = _time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for idx, input_item, _stage1_result in llm_queue:
                future = executor.submit(
                    judge,
                    input_item.id,
                    input_item.user_question,
                    input_item.system_reply,
                    input_item.knowledge_base,
                )
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                result = stage1_results[idx]
                _t_one = _time.time()
                llm_result = future.result()
                _dt_one = _time.time() - _t_one

                print(f"  LLM {result.id}: result={llm_result['is_hallucination']} (got in {_dt_one:.0f}s, total wait)")

                if llm_result["is_hallucination"] is True:
                    result.is_hallucination = True
                    result.detection_layer = DetectionLayer.L3_UNSUPPORTED_CLAIM
                    result.confidence = Confidence(llm_result.get("confidence", "中"))
                    result.reason = llm_result["reason"]
                    result = classify_llm_result(result, llm_result.get("subtype", ""))
                elif llm_result["is_hallucination"] is False:
                    result.is_hallucination = False
                    result.confidence = Confidence(llm_result.get("confidence", "中"))
                    result.reason = llm_result["reason"]
                    result.detection_layer = None
                    result.output_type = None

                stage1_results[idx] = result

        _t_llm = _time.time() - _t_llm_start
        _t_total = _time.time() - _t0
        print(f"  LLM phase done: {_t_llm:.1f}s | Total: {_t_total:.1f}s")

    return stage1_results


def run_and_save(
    input_file: str = "data/replies.json",
    output_file: str = "data/detection_results.json",
) -> list[DetectionResult]:
    """Load replies, run detection, save results to JSON and SQLite."""
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / input_file
    output_path = base_dir / output_file

    with open(input_path, encoding="utf-8") as f:
        replies = json.load(f)

    print(f"Running hallucination detection on {len(replies)} replies...")
    results = detect_batch(replies)

    # Save to JSON
    output_data = [r.model_dump() for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")

    # Save to SQLite
    _save_to_db(replies, results, input_file)

    # Print summary
    hallucination_count = sum(1 for r in results if r.is_hallucination)
    non_hallucination_count = sum(1 for r in results if r.is_hallucination is False)
    uncertain_count = sum(1 for r in results if r.is_hallucination is None)
    print(f"\nSummary: {hallucination_count} hallucinations, "
          f"{non_hallucination_count} non-hallucinations, "
          f"{uncertain_count} uncertain (out of {len(results)})")

    return results


def _save_to_db(replies: list[dict], results: list[DetectionResult], filename: str) -> None:
    init_db()
    conn = get_connection()
    try:
        now = datetime.now().isoformat()
        hallucination_count = sum(1 for r in results if r.is_hallucination)

        cursor = conn.execute(
            "INSERT INTO detection_batches (filename, total_count, hallucination_count, created_at) VALUES (?, ?, ?, ?)",
            (filename, len(results), hallucination_count, now),
        )
        batch_id = cursor.lastrowid

        for reply, result in zip(replies, results):
            conn.execute(
                """INSERT INTO detection_results
                   (batch_id, reply_id, user_question, system_reply, knowledge_base,
                    is_hallucination, detection_layer, output_type, confidence, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    batch_id,
                    result.id,
                    reply["user_question"],
                    reply["system_reply"],
                    reply["knowledge_base"],
                    1 if result.is_hallucination is True else (0 if result.is_hallucination is False else None),
                    result.detection_layer.value if result.detection_layer else None,
                    result.output_type.value if result.output_type else None,
                    result.confidence.value,
                    result.reason,
                    now,
                ),
            )

        conn.commit()
        print(f"Saved to SQLite: batch_id={batch_id}")
    finally:
        conn.close()
