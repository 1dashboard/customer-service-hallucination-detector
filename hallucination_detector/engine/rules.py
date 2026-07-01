"""Stage 1 Rule Engine: L1 (direct contradiction) and L2 (capability overreach) detection."""

from __future__ import annotations

import re
from typing import Optional

from .models import Confidence, DetectionLayer, DetectionResult

# ── KB negation patterns (L2) ──────────────────────────────────────────
KB_NEGATION_PATTERNS = [
    re.compile(r"未接入"),
    re.compile(r"不具备"),
    re.compile(r"不可(?!以)"),  # 不可告知, not 不可以
    re.compile(r"无（"),
    re.compile(r"不支持"),
    re.compile(r"暂不支持"),
    re.compile(r"不允许"),
    re.compile(r"无线下门店"),
    re.compile(r"无线下"),
    re.compile(r"未标注"),
    re.compile(r"未提及"),
    re.compile(r"无此"),
    re.compile(r"当前无"),
]

# Reply action verbs (indicates the system claims it DID something)
REPLY_ACTION_PATTERNS = [
    re.compile(r"已帮您|已经帮您|已为您"),
    re.compile(r"我帮您查了|我帮您查到了"),
    re.compile(r"已修改|已升级|已处理"),
    re.compile(r"发到您账户|直接发到|发到账户"),
    re.compile(r"包裹目前在|目前在.*转运"),
    re.compile(r"退到|退款已经在"),
    re.compile(r"会有专属客服"),
]


def _extract_key_value_pairs(text: str) -> dict[str, str]:
    """Extract (category, value) pairs from text using regex patterns.

    Returns dict where keys are category names and values are extracted values.
    """
    pairs: dict[str, str] = {}

    # Return/delivery days: "X天(无理由退货|内可退换)"
    m = re.search(r"(\d+)\s*天\s*无理由", text)
    if m:
        pairs["无理由退货天数"] = m.group(1)

    m = re.search(r"(\d+)\s*天\s*内\s*可退换", text)
    if m:
        pairs["质量退货天数"] = m.group(1)

    # Shipping time: "X小时内发货"
    m = re.search(r"(\d+)\s*小时\s*内?\s*发货", text)
    if m:
        pairs["发货时间_小时"] = m.group(1)

    # Bluetooth version
    m = re.search(r"蓝牙\s*(\d+(?:\.\d+)?)", text)
    if m:
        pairs["蓝牙版本"] = m.group(1)

    # Latency
    m = re.search(r"延迟[约低至]*\s*(\d+)\s*ms", text)
    if m:
        pairs["延迟_ms"] = m.group(1)

    # Interface type — prefer the one near "接口" or "输出" (actual product spec)
    m = re.search(r"接口[类型]*[：:]\s*(USB-A|USB-C|Type-C|Lightning|Micro-USB)", text)
    if not m:
        m = re.search(r"(USB-A|USB-C|Type-C|Lightning|Micro-USB)\s*输出", text)
    if m:
        pairs["接口类型"] = m.group(1)
    else:
        # Fallback: first interface found in text
        for interface in ["USB-A", "USB-C", "Type-C", "Lightning", "Micro-USB"]:
            if interface in text:
                pairs["接口类型"] = interface
                break

    # Material
    for mat in ["头层牛皮", "二层牛皮", "PU合成革", "真皮", "纯棉"]:
        if mat in text:
            pairs["材质"] = mat
            break

    # Warranty
    m = re.search(r"保修期?\s*(?:为|：)?\s*(\d+)\s*(?:年|个?月)", text)
    if m:
        pairs["保修期"] = m.group(1)

    # Express company
    for exp in ["顺丰", "中通", "韵达", "圆通", "申通", "京东", "EMS", "德邦"]:
        if exp in text:
            pairs["快递公司"] = exp
            break

    # Invoice type
    if "纸质发票" in text:
        pairs["纸质发票"] = "支持"
    if "电子发票" in text:
        pairs["电子发票"] = "支持"
    m = re.search(r"不支持\s*纸质发票", text)
    if m:
        pairs["纸质发票"] = "不支持"
    m = re.search(r"暂不支持\s*纸质发票", text)
    if m:
        pairs["纸质发票"] = "不支持"

    # Coupon amounts: 满X减Y
    m = re.search(r"满\s*(\d+)\s*减\s*(\d+)", text)
    if m:
        pairs["满减优惠"] = f"{m.group(1)}-{m.group(2)}"

    # Discount: X折
    m = re.search(r"(\d+(?:\.\d+)?)\s*折", text)
    if m:
        pairs["折扣"] = m.group(1)

    # Shipping days
    m = re.search(r"(\d+)\s*[-~]\s*(\d+)\s*天\s*(?:到|送|可)", text)
    if m:
        pairs["到货天数"] = f"{m.group(1)}-{m.group(2)}"

    # Has offline stores
    m = re.search(r"无线下|纯线上", text)
    if m:
        pairs["线下门店"] = "无"

    m = re.search(r"线下(?:体验|门店|实体)店", text)
    if m:
        pairs["线下门店"] = "有"

    # Payment methods - cash on delivery
    if "货到付款" in text:
        m = re.search(r"(?:不支持|暂不支持)\s*货到付款", text)
        if m:
            pairs["货到付款"] = "不支持"
        else:
            pairs["货到付款"] = "支持"

    # Student discount
    if "学生" in text:
        m = re.search(r"无学生优惠|无.*学生.*优惠", text)
        if m:
            pairs["学生优惠"] = "无"
        elif "学生" in text and ("优惠" in text or "折扣" in text or "折" in text):
            pairs["学生优惠"] = "有"

    # Safety - pregnancy
    if "孕妇" in text:
        if "可以放心" in text or "可以使用" in text or "安全" in text:
            pairs["孕妇可用"] = "是"
        elif "咨询医生" in text or "建议咨询" in text:
            pairs["孕妇可用"] = "咨询医生"

    # Shoe size info
    m = re.search(r"偏大|偏小|偏大半码|偏小半码", text)
    if m:
        pairs["尺码偏差"] = m.group()
    if "尺码标准" in text and "不偏" in text:
        pairs["尺码偏差"] = "标准"

    return pairs


def detect_l1(kb: str, reply: str) -> Optional[dict]:
    """Detect L1: direct contradiction between KB and reply.

    Compares key-value pairs extracted from both texts.
    """
    kb_pairs = _extract_key_value_pairs(kb)
    reply_pairs = _extract_key_value_pairs(reply)

    contradictions = []

    for key, kb_val in kb_pairs.items():
        reply_val = reply_pairs.get(key)
        if reply_val is not None and reply_val != kb_val:
            contradictions.append({
                "key": key,
                "kb_value": kb_val,
                "reply_value": reply_val,
            })

    if contradictions:
        return {"contradictions": contradictions}

    # Additional check: entities present in reply but absent/contradicted in KB
    for key, reply_val in reply_pairs.items():
        if key not in kb_pairs:
            # Reply contains a claim that KB doesn't mention at all
            # Skip for now, let LLM handle via L3
            pass

    return None


def detect_l2(kb: str, reply: str) -> Optional[dict]:
    """Detect L2: capability overreach - KB says no capability, reply claims execution."""
    has_negation = any(p.search(kb) for p in KB_NEGATION_PATTERNS)
    has_action = any(p.search(reply) for p in REPLY_ACTION_PATTERNS)

    if has_negation and has_action:
        domain = _extract_negation_domain(kb)
        return {
            "kb_negation": True,
            "reply_claims_action": True,
            "domain": domain or "未知能力",
        }

    # Additional L2 check: KB explicitly says something doesn't exist, reply says it does
    # e.g., KB: "无线下门店" → Reply: "我们在北京、上海有体验店"
    kb_denies_existence = bool(re.search(
        r"无线下门店|纯线上|无此|当前无|无.*优惠|无.*活动",
        kb,
    ))
    reply_claims_existence = bool(re.search(
        r"有的[，,]|我们在|线下体验店|门店查询|学生认证|满\d+减\d+",
        reply,
    ))

    if kb_denies_existence and reply_claims_existence:
        return {
            "kb_negation": True,
            "reply_claims_action": True,
            "domain": "KB明确否认但回复声称存在",
        }

    return None


def _extract_negation_domain(kb: str) -> str:
    """Extract what capability the KB says the system doesn't have."""
    for pattern in KB_NEGATION_PATTERNS:
        m = pattern.search(kb)
        if m:
            rest = kb[m.end():]
            chars = re.findall(r"[一-鿿]{2,8}", rest)
            if chars:
                return chars[0]
    return ""


def run_stage1(reply_id: str, user_question: str, system_reply: str, knowledge_base: str) -> DetectionResult:
    """Run Stage 1 rule engine on a single reply."""
    l1_result = detect_l1(knowledge_base, system_reply)
    l2_result = detect_l2(knowledge_base, system_reply)

    if l1_result and l1_result.get("contradictions"):
        contradictions = l1_result["contradictions"]
        reasons = []
        for c in contradictions:
            reasons.append(f"[{c['key']}] KB:{c['kb_value']} → Reply:{c['reply_value']}")

        return DetectionResult(
            id=reply_id,
            is_hallucination=True,
            detection_layer=DetectionLayer.L1_DIRECT_CONTRADICTION,
            confidence=Confidence.HIGH,
            reason="; ".join(reasons),
            output_type=None,
        )

    if l2_result:
        return DetectionResult(
            id=reply_id,
            is_hallucination=True,
            detection_layer=DetectionLayer.L2_CAPABILITY_OVERREACH,
            confidence=Confidence.HIGH,
            reason=f"KB声明无[{l2_result['domain']}]能力，但回复声称已执行相关操作",
        )

    # Pre-filter for L3: KB has no info / negation but reply is affirmative
    kb_has_no_info_or_denial = bool(re.search(
        r"无（|未标注|未提及|当前无|无此|无线下",
        knowledge_base,
    ))
    reply_is_affirmative = bool(re.search(r"是的|有的|支持的|可以的|我们在|凭.*可以", system_reply))

    if kb_has_no_info_or_denial and reply_is_affirmative:
        return DetectionResult(
            id=reply_id,
            is_hallucination=True,
            detection_layer=DetectionLayer.L3_UNSUPPORTED_CLAIM,
            confidence=Confidence.LOW,
            reason="KB无此信息或明确否认，reply给出肯定答复，需LLM进一步判定",
        )

    # KB has no useful info → potential L3
    kb_is_empty = len(knowledge_base) < 20 or "无（" in knowledge_base
    if kb_is_empty:
        return DetectionResult(
            id=reply_id,
            is_hallucination=None,
            detection_layer=DetectionLayer.L3_UNSUPPORTED_CLAIM,
            confidence=Confidence.LOW,
            reason="需LLM进行语义级比对",
        )

    # KB has specific info, rules didn't find contradiction → likely not hallucination
    return DetectionResult(
        id=reply_id,
        is_hallucination=None,
        detection_layer=None,
        confidence=Confidence.LOW,
        reason="规则引擎未检测到明确矛盾，建议LLM二次确认",
    )
