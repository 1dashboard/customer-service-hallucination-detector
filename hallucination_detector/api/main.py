"""FastAPI backend for hallucination detection service."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Add parent dir to path for engine imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from engine.db import get_connection, init_db
from engine.llm_judge import get_client
from engine.models import DetectionInput, DetectionResult
from engine.pipeline import detect_batch, detect_single

app = FastAPI(title="Hallucination Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    client = get_client()
    return {
        "status": "ok",
        "llm_available": client is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/detect/single")
def detect_single_endpoint(item: DetectionInput) -> DetectionResult:
    """Run detection on a single reply item."""
    try:
        result = detect_single(item)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/batch")
def detect_batch_endpoint(items: list[dict]) -> list[dict]:
    """Run detection on a batch of reply items."""
    try:
        results = detect_batch(items)
        return [r.model_dump() for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/upload")
async def detect_upload_endpoint(file: UploadFile = File(...)) -> dict:
    """Upload a JSON file containing reply items and run detection."""
    try:
        content = await file.read()
        items = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="JSON must be an array of reply items")

    # Validate required fields
    for i, item in enumerate(items):
        for field in ("user_question", "system_reply", "knowledge_base"):
            if field not in item:
                raise HTTPException(
                    status_code=422,
                    detail=f"Item at index {i} is missing required field: {field}",
                )

    results = detect_batch(items)
    results_data = [r.model_dump() for r in results]

    # Save to DB
    init_db()
    conn = get_connection()
    try:
        now = datetime.now().isoformat()
        hallucination_count = sum(1 for r in results if r.is_hallucination)
        cursor = conn.execute(
            "INSERT INTO detection_batches (filename, total_count, hallucination_count, created_at) VALUES (?, ?, ?, ?)",
            (file.filename or "upload", len(results), hallucination_count, now),
        )
        batch_id = cursor.lastrowid

        for item, result in zip(items, results):
            conn.execute(
                """INSERT INTO detection_results
                   (batch_id, reply_id, user_question, system_reply, knowledge_base,
                    is_hallucination, detection_layer, output_type, confidence, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    batch_id,
                    result.id,
                    item["user_question"],
                    item["system_reply"],
                    item["knowledge_base"],
                    1 if result.is_hallucination is True else (0 if result.is_hallucination is False else None),
                    result.detection_layer.value if result.detection_layer else None,
                    result.output_type.value if result.output_type else None,
                    result.confidence.value,
                    result.reason,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "batch_id": batch_id,
        "filename": file.filename,
        "total": len(results),
        "hallucination_count": hallucination_count,
        "results": results_data,
    }


@app.get("/api/results")
def get_results(
    batch_id: Optional[int] = Query(None),
    output_type: Optional[str] = Query(None),
    is_hallucination: Optional[bool] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    """Query historical detection results with filtering and pagination."""
    conn = get_connection()
    try:
        conditions = []
        params: list = []

        if batch_id is not None:
            conditions.append("batch_id = ?")
            params.append(batch_id)
        if output_type is not None:
            conditions.append("output_type = ?")
            params.append(output_type)
        if is_hallucination is not None:
            conditions.append("is_hallucination = ?")
            params.append(1 if is_hallucination else 0)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"SELECT COUNT(*) as total FROM detection_results {where}"
        total = conn.execute(count_sql, params).fetchone()["total"]

        offset = (page - 1) * page_size
        sql = f"SELECT * FROM detection_results {where} ORDER BY id DESC LIMIT ? OFFSET ?"
        rows = conn.execute(sql, params + [page_size, offset]).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "batch_id": row["batch_id"],
                "reply_id": row["reply_id"],
                "user_question": row["user_question"],
                "system_reply": row["system_reply"],
                "knowledge_base": row["knowledge_base"],
                "is_hallucination": bool(row["is_hallucination"]) if row["is_hallucination"] is not None else None,
                "detection_layer": row["detection_layer"],
                "output_type": row["output_type"],
                "confidence": row["confidence"],
                "reason": row["reason"],
                "created_at": row["created_at"],
            })

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "results": results,
        }
    finally:
        conn.close()


@app.post("/api/evaluate")
def evaluate_endpoint(request: dict) -> dict:
    """Compare detection results with ground truth and compute metrics."""
    ground_truth: list[dict] = request.get("ground_truth", [])
    results: list[dict] = request.get("results", [])

    if not ground_truth or not results:
        raise HTTPException(status_code=400, detail="Both ground_truth and results are required")

    from engine.evaluator import compute_metrics

    metrics = compute_metrics(ground_truth, results)
    return metrics.model_dump()
