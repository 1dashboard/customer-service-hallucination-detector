"""Classification mapper: detection layer (L1/L2/L3) → output layer (business labels)."""

from __future__ import annotations

from .models import DetectionLayer, DetectionResult, OutputType


def classify(result: DetectionResult, user_question: str = "", knowledge_base: str = "") -> DetectionResult:
    """Map detection layer result to business-facing output type.

    Mapping rules:
    - L1 → policy_fabrication / policy_deviation / param_fabrication (context-dependent)
    - L2 → capability_overreach
    - L3 → promotion_fabrication / info_fabrication / safety_misleading / info_omission
    """
    if result.detection_layer is None:
        return result

    layer = result.detection_layer

    if layer == DetectionLayer.L1_DIRECT_CONTRADICTION:
        result.output_type = _classify_l1(result.reason, knowledge_base, user_question)

    elif layer == DetectionLayer.L2_CAPABILITY_OVERREACH:
        result.output_type = OutputType.CAPABILITY_OVERREACH

    elif layer == DetectionLayer.L3_UNSUPPORTED_CLAIM:
        result.output_type = _classify_l3(result.reason, knowledge_base, user_question)

    return result


def classify_llm_result(result: DetectionResult, llm_subtype: str = "") -> DetectionResult:
    """Apply classification after LLM has provided a subtype hint.

    Args:
        result: DetectionResult with detection_layer = L3
        llm_subtype: String hint from LLM output (e.g., 'safety_misleading', 'promotion_fabrication')
    """
    if not llm_subtype:
        result.output_type = OutputType.INFO_FABRICATION
        return result

    subtype_map = {
        "安全误导": OutputType.SAFETY_MISLEADING,
        "优惠编造": OutputType.PROMOTION_FABRICATION,
        "信息编造": OutputType.INFO_FABRICATION,
        "信息遗漏": OutputType.INFO_OMISSION,
        "政策偏差": OutputType.POLICY_DEVIATION,
        "政策编造": OutputType.POLICY_FABRICATION,
        "参数编造": OutputType.PARAM_FABRICATION,
        "能力越界": OutputType.CAPABILITY_OVERREACH,
        # Also accept LLM output subtypes
        "直接矛盾": OutputType.PARAM_FABRICATION,
        "凭空编造": OutputType.INFO_FABRICATION,
    }
    result.output_type = subtype_map.get(llm_subtype, OutputType.INFO_FABRICATION)
    return result


def _classify_l1(reason: str, kb: str, question: str) -> OutputType:
    kb_lower = kb.lower()
    question_lower = question.lower()
    reason_lower = reason.lower()

    # Safety-related (highest priority, even in L1)
    safety_keywords = ["孕妇", "儿童", "婴儿", "过敏", "安全", "毒性", "副作用", "视黄醇", "哺乳"]
    if any(k in kb_lower for k in safety_keywords):
        return OutputType.SAFETY_MISLEADING

    # Policy-related
    policy_keywords = ["退货", "退款", "发货", "发票", "运费", "质保", "保修", "售后", "申请", "政策", "学生"]
    if any(k in kb_lower for k in policy_keywords):
        if "天" in reason or "小时" in reason:
            return OutputType.POLICY_DEVIATION
        return OutputType.POLICY_FABRICATION

    # Product parameters
    param_keywords = ["蓝牙", "接口", "材质", "NFC", "尺寸", "重量", "容量", "电池", "参数", "规格", "颜色", "尺码", "门店"]
    if any(k in kb_lower or k in question_lower for k in param_keywords):
        return OutputType.PARAM_FABRICATION

    # Default for L1
    return OutputType.PARAM_FABRICATION


def _classify_l3(reason: str, kb: str, question: str) -> OutputType:
    kb_lower = kb.lower()
    reason_lower = reason.lower()

    # Safety-related (highest priority)
    safety_keywords = ["孕妇", "儿童", "婴儿", "过敏", "安全", "毒性", "副作用", "视黄醇", "哺乳"]
    if any(k in kb_lower for k in safety_keywords):
        return OutputType.SAFETY_MISLEADING

    # Promotion/coupon
    promo_keywords = ["优惠", "满减", "折扣", "学生", "优惠券", "活动", "满\\d+减"]
    if any(k in kb_lower for k in promo_keywords) or "优惠" in reason_lower or "折" in reason_lower:
        return OutputType.PROMOTION_FABRICATION

    # Information omission
    omission_keywords = ["遗漏", "不完整", "偏大", "偏小", "只说", "未提及"]
    if any(k in reason_lower for k in omission_keywords):
        return OutputType.INFO_OMISSION

    # Default for L3
    return OutputType.INFO_FABRICATION
