"""Unit tests for LLM judge (with mock responses)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.llm_judge import _parse_llm_output, _fallback, judge


def test_parse_json_output() -> None:
    """Test parsing clean JSON from LLM."""
    content = '{"is_hallucination": true, "subtype": "contradiction", "reason": "KB says 7 days, reply says 30 days", "confidence": "HIGH"}'
    result = _parse_llm_output(content)

    assert result["is_hallucination"] is True
    assert result["subtype"] == "contradiction"
    assert result["confidence"] == "HIGH"
    print("  [PASS] test_parse_json_output")


def test_parse_markdown_fenced_output() -> None:
    """Test parsing JSON inside markdown code fences."""
    content = '```json\n{"is_hallucination": false, "subtype": "none", "reason": "Content matches", "confidence": "HIGH"}\n```'
    result = _parse_llm_output(content)

    assert result["is_hallucination"] is False
    assert result["subtype"] == "none"
    print("  [PASS] test_parse_markdown_fenced_output")


def test_parse_invalid_json() -> None:
    """Test fallback on invalid JSON."""
    content = "This is not JSON at all"
    result = _parse_llm_output(content)

    assert result["is_hallucination"] is None
    assert "Failed to parse" in result["reason"]
    print("  [PASS] test_parse_invalid_json")


def test_fallback() -> None:
    """Test the fallback result structure."""
    result = _fallback("Test error message")

    assert result["is_hallucination"] is None
    assert result["confidence"] == "LOW"
    assert result["reason"] == "Test error message"
    print("  [PASS] test_fallback")


@patch("engine.llm_judge.get_client")
def test_judge_without_api_key(mock_get_client: MagicMock) -> None:
    """Test judge returns fallback when no API key configured."""
    mock_get_client.return_value = None

    result = judge("test_01", "question?", "reply text", "KB text")
    assert result["is_hallucination"] is None
    assert "not configured" in result["reason"]
    print("  [PASS] test_judge_without_api_key")


def test_parse_partial_json() -> None:
    """Test extracting JSON embedded in other text."""
    content = 'Here is my analysis: {"is_hallucination": true, "subtype": "fabrication", "reason": "Made up info", "confidence": "MEDIUM"} End.'
    result = _parse_llm_output(content)

    assert result["is_hallucination"] is True
    assert result["subtype"] == "fabrication"
    print("  [PASS] test_parse_partial_json")


def run_all() -> None:
    print("Running LLM Judge Tests...\n")
    tests = [
        test_parse_json_output,
        test_parse_markdown_fenced_output,
        test_parse_invalid_json,
        test_fallback,
        test_judge_without_api_key,
        test_parse_partial_json,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
