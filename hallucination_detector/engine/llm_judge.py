"""Stage 2 LLM Judge: DeepSeek v3 API for semantic-level hallucination detection."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

SYSTEM_PROMPT = """你是一个客服回复幻觉检测专家。你的任务是比对【知识库内容】和【客服系统回复】，判断客服的回复是否存在"幻觉"。

## 幻觉类型定义
- **直接矛盾**: 知识库有明确信息，回复改成了不同的值（如数字、名称、政策条款）
- **凭空编造**: 知识库中无相关信息或明确没有，回复却肯定地声称有
- **信息遗漏**: 知识库有关键的限制条件或风险提示，回复忽略了这些信息导致误导
- **安全误导**: 知识库有安全/健康相关的风险提示，回复却说"安全"或"可以放心"
- **政策偏差**: 回复部分正确但部分错误，或声称的政策与知识库不完全一致
- **无幻觉**: 回复与知识库一致，或措辞不同但语义相同

## 输出格式
你必须严格输出一个 JSON 对象，包含以下字段：
{
  "is_hallucination": true/false,
  "subtype": "直接矛盾" | "凭空编造" | "信息遗漏" | "安全误导" | "政策偏差" | "无幻觉",
  "reason": "用中文简要说明判定依据",
  "confidence": "高" | "中" | "低"
}

## 注意事项
1. 如果回复内容在知识库中完全找不到依据，即使听起来合理，也要判为幻觉
2. 如果知识库说"无此功能/未接入/不支持"，而回复声称提供了该功能，判为编造
3. 涉及孕妇、儿童、过敏等安全相关的判定要格外严格
4. 如果回复和知识库只是措辞不同但语义一致，不判为幻觉
"""


def get_client() -> Optional[OpenAI]:
    if not DEEPSEEK_API_KEY:
        return None
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=30.0,
    )


def judge(
    reply_id: str,
    user_question: str,
    system_reply: str,
    knowledge_base: str,
    max_retries: int = 3,
) -> dict:
    """Send KB + reply to DeepSeek for semantic hallucination judgment.

    Returns a dict with keys: is_hallucination, subtype, reason, confidence.
    On persistent failure, returns is_hallucination=null.
    """
    client = get_client()
    if client is None:
        return _fallback("DeepSeek API key not configured")

    user_prompt = f"""## 用户问题
{user_question}

## 知识库内容
{knowledge_base}

## 客服系统回复
{system_reply}

请判断客服回复是否存在幻觉，输出 JSON。"""

    last_error = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or "{}"
            return _parse_llm_output(content)

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)

    return _fallback(f"LLM API call failed after {max_retries} attempts: {last_error}")


def judge_batch(
    items: list[dict],
) -> list[dict]:
    """Run LLM judgment on multiple items sequentially.

    Each item should have: id, user_question, system_reply, knowledge_base.
    """
    results = []
    for item in items:
        result = judge(
            reply_id=item.get("id", ""),
            user_question=item["user_question"],
            system_reply=item["system_reply"],
            knowledge_base=item["knowledge_base"],
        )
        result["id"] = item.get("id", "")
        results.append(result)
    return results


def _parse_llm_output(content: str) -> dict:
    """Parse JSON from LLM output, handling markdown code blocks."""
    content = content.strip()

    # Remove markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        parsed = json.loads(content)
        return {
            "is_hallucination": parsed.get("is_hallucination", False),
            "subtype": parsed.get("subtype", "none"),
            "reason": parsed.get("reason", ""),
            "confidence": parsed.get("confidence", "中"),
        }
    except json.JSONDecodeError:
        # Try to extract JSON from the content
        import re
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {
                    "is_hallucination": parsed.get("is_hallucination", False),
                    "subtype": parsed.get("subtype", "none"),
                    "reason": parsed.get("reason", ""),
                    "confidence": parsed.get("confidence", "中"),
                }
            except json.JSONDecodeError:
                pass
        return _fallback(f"Failed to parse LLM output: {content[:200]}")


def _fallback(reason: str) -> dict:
    return {
        "is_hallucination": None,
        "subtype": "none",
        "reason": reason,
        "confidence": "低",
    }
