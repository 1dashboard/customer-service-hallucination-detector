#!/usr/bin/env python
"""One-click startup: launches FastAPI (port 8000) + Streamlit (port 8501)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent

    print("=" * 50)
    print("  Hallucination Detector - Starting Services")
    print("=" * 50)

    # Start FastAPI
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=str(root),
    )
    print("  [API]  FastAPI starting on http://localhost:8000")
    print("  [API]  Docs:   http://localhost:8000/docs")
    time.sleep(2)

    # Start Streamlit
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", "8501"],
        cwd=str(root),
    )
    print("  [UI]   Streamlit starting on http://localhost:8501")
    print("=" * 50)
    print("  Press Ctrl+C to stop all services.")
    print("=" * 50)

    try:
        api_proc.wait()
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_proc.terminate()
        streamlit_proc.terminate()
        api_proc.wait()
        streamlit_proc.wait()
        print("All services stopped.")


if __name__ == "__main__":
    main()
