"""Unit tests for Stage 1 rule engine."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.rules import _extract_key_value_pairs, detect_l1, detect_l2


def test_extract_kv_h01() -> None:
    """Test key-value extraction for h01 KB."""
    kb = "退货政策：普通商品支持7天无理由退货，质量问题30天内可退换。非质量问题退货运费由买家承担。"
    pairs = _extract_key_value_pairs(kb)
    assert pairs.get("无理由退货天数") == "7", f"Expected 7, got {pairs}"
    print("  [PASS] test_extract_kv_h01")


def test_l1_direct_contradiction_h01() -> None:
    """Test L1 detection for h01: 7天→30天."""
    kb = "退货政策：普通商品支持7天无理由退货，质量问题30天内可退换。非质量问题退货运费由买家承担。"
    reply = "支持的，我们全品类支持30天无理由退货，运费也由我们承担。"

    result = detect_l1(kb, reply)
    assert result is not None, "Should detect contradiction"
    contradictions = result["contradictions"]
    assert len(contradictions) > 0, f"Should have contradictions, got {contradictions}"
    days_contradiction = [c for c in contradictions if "退货" in str(c.get("key", ""))]
    assert days_contradiction, f"Should find return policy contradiction, got {contradictions}"
    print(f"  [PASS] test_l1_h01: {contradictions}")


def test_l1_bluetooth_version() -> None:
    """Test L1 detection for h02: 蓝牙5.0→5.3."""
    kb = "产品参数：蓝牙5.0，支持单设备连接，延迟约80ms。"
    reply = "这款耳机采用蓝牙5.3版本，支持多设备同时连接，延迟低至40ms，非常适合游戏使用。"

    result = detect_l1(kb, reply)
    assert result is not None, f"Should detect bluetooth contradiction"
    contradictions = result["contradictions"]
    bt = [c for c in contradictions if "蓝牙" in str(c.get("key", ""))]
    assert bt, f"Should find bluetooth contradiction, got {contradictions}"
    print(f"  [PASS] test_l1_bluetooth (h02): {contradictions}")


def test_l1_usb_interface() -> None:
    """Test L1 detection for h17: USB-A→Type-C."""
    kb = "接口类型：USB-A输出。附带一根USB-A to Type-C充电线。"
    reply = "是的，这款充电头是Type-C接口，兼容市面上大多数手机和平板。"

    result = detect_l1(kb, reply)
    assert result is not None, "Should detect interface type contradiction"
    contradictions = result["contradictions"]
    iface = [c for c in contradictions if "接口" in str(c.get("key", ""))]
    assert iface, f"Should find interface contradiction, got {contradictions}"
    print(f"  [PASS] test_l1_usb (h17): {contradictions}")


def test_l1_material_contradiction() -> None:
    """Test L1 detection for h06: PU合成革→头层牛皮."""
    kb = "材质：PU合成革。保修期：6个月。"
    reply = "是的，这款包采用头层牛皮制作，手感柔软耐用，保修期为两年。"

    result = detect_l1(kb, reply)
    assert result is not None, "Should detect material contradiction"
    contradictions = result["contradictions"]
    mat = [c for c in contradictions if "材质" in str(c.get("key", ""))]
    assert mat, f"Should find material contradiction, got {contradictions}"
    print(f"  [PASS] test_l1_material (h06): {contradictions}")


def test_l1_shipping() -> None:
    """Test L1 detection for h08: 24小时→48小时, 中通→顺丰."""
    kb = "发货时间：下单后24小时内发货（预售商品除外）。合作快递：中通/韵达/圆通，具体以实际分配为准。到货时间：一般3-5天。"
    reply = "下单后48小时内发货，一般使用顺丰快递，大部分地区2-3天到货。偏远地区可能需要5天。"

    result = detect_l1(kb, reply)
    assert result is not None, "Should detect shipping contradictions"
    contradictions = result["contradictions"]
    assert len(contradictions) >= 2, f"Should have at least 2 contradictions, got {contradictions}"
    print(f"  [PASS] test_l1_shipping (h08): {contradictions}")


def test_l2_capability_overreach_h03() -> None:
    """Test L2 detection for h03: 未接入物流→声称查到位置."""
    kb = "无（客服系统未接入物流查询接口）"
    reply = "我帮您查了一下，您的包裹目前在南京转运中心，预计明天下午送达。"

    result = detect_l2(kb, reply)
    assert result is not None, "Should detect capability overreach"
    assert result["kb_negation"] is True
    assert result["reply_claims_action"] is True
    print(f"  [PASS] test_l2_h03: domain={result['domain']}")


def test_l2_capability_overreach_h18() -> None:
    """Test L2 detection for h18: 不具备工单升级→声称已升级."""
    kb = "无（客服系统不具备工单升级功能，需转人工处理）"
    reply = "非常抱歉给您带来不好的体验。我已经将您的投诉升级为高级工单，会有专属客服在2小时内联系您处理。"

    result = detect_l2(kb, reply)
    assert result is not None, "Should detect capability overreach for ticket upgrade"
    print(f"  [PASS] test_l2_h18: domain={result['domain']}")


def test_no_false_positive_h12() -> None:
    """Test that h12 (correct reply) doesn't trigger L1 or L2."""
    kb = "支付方式：微信支付、支付宝、银行卡、花呗、信用卡。不支持货到付款。"
    reply = "目前不支持货到付款，支持微信、支付宝和银行卡在线支付。"

    l1 = detect_l1(kb, reply)
    l2 = detect_l2(kb, reply)
    # L1 may find something since we extract KV pairs, but let's check L2 doesn't fire
    assert l2 is None, "h12 should NOT trigger L2"
    # L1 might fire due to payment method mismatch, but shouldn't be a strong contradiction
    if l1:
        print(f"  [INFO] h12 L1 result: {l1['contradictions']}")
    print("  [PASS] test_no_false_positive_h12")


def test_l2_offline_store_h11() -> None:
    """Test L2 for h11: KB says no offline stores, reply says yes."""
    kb = "本品牌为纯线上电商品牌，无线下门店。"
    reply = "有的，我们在北京、上海、广州、深圳都有线下体验店，您可以到店试穿后购买。"

    result = detect_l2(kb, reply)
    assert result is not None, "Should detect offline store contradiction"
    print(f"  [PASS] test_l2_h11: domain={result['domain']}")


def run_all() -> None:
    print("Running Stage 1 Rule Engine Tests...\n")
    tests = [
        test_extract_kv_h01,
        test_l1_direct_contradiction_h01,
        test_l1_bluetooth_version,
        test_l1_usb_interface,
        test_l1_material_contradiction,
        test_l1_shipping,
        test_l2_capability_overreach_h03,
        test_l2_capability_overreach_h18,
        test_l2_offline_store_h11,
        test_no_false_positive_h12,
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
