from __future__ import annotations

import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "detection.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS detection_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            total_count INTEGER NOT NULL DEFAULT 0,
            hallucination_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            reply_id TEXT NOT NULL,
            user_question TEXT NOT NULL,
            system_reply TEXT NOT NULL,
            knowledge_base TEXT NOT NULL,
            is_hallucination INTEGER,
            detection_layer TEXT,
            output_type TEXT,
            confidence TEXT NOT NULL DEFAULT 'LOW',
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES detection_batches(id)
        );

        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1 REAL NOT NULL,
            true_positives INTEGER NOT NULL,
            true_negatives INTEGER NOT NULL,
            false_positives INTEGER NOT NULL,
            false_negatives INTEGER NOT NULL,
            details TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES detection_batches(id)
        );
        """)
        conn.commit()
    finally:
        conn.close()
