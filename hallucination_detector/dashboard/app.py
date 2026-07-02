"""Streamlit dashboard for hallucination detection results."""

from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db import get_connection, init_db
from engine.evaluator import compute_metrics

st.set_page_config(
    page_title="客服幻觉检测系统",
    page_icon="🔍",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# ── Theme palettes ─────────────────────────────────────────────────

DARK = {
    'bg': '#0D1117', 'bg2': '#161B22', 'border': '#30363D', 'border2': '#21262D',
    'text': '#E6EDF3', 'text2': '#C9D1D9', 'text3': '#8B949E', 'text4': '#6E7681',
    'primary': '#4F8CFF', 'success': '#3FB950', 'danger': '#F85149', 'warning': '#D29922',
    'purple': '#A371F7', 'headerBg': '#1C2333', 'cardShadow': 'rgba(0,0,0,0.3)',
    'successBg': 'rgba(63,185,80,0.12)', 'dangerBg': 'rgba(248,81,73,0.12)',
    'warningBg': 'rgba(210,153,34,0.12)', 'primaryBg': 'rgba(79,140,255,0.12)',
    'scrollbarTrack': '#0D1117', 'scrollbarThumb': '#30363D', 'scrollbarThumbHover': '#484F58',
    'btnShadow': 'rgba(79,140,255,0.4)', 'btnHoverShadow': 'rgba(79,140,255,0.6)',
    'btnGradientStart': '#4F8CFF', 'btnGradientEnd': '#3A6FD8',
    'btnSecondaryBg': '#21262D', 'primaryHoverBg': 'rgba(79,140,255,0.04)',
    'primaryHoverBg2': 'rgba(79,140,255,0.06)', 'primaryHoverBg3': 'rgba(79,140,255,0.03)',
    'primaryHoverBg4': 'rgba(79,140,255,0.08)',
    'sidebarBg': '#0D1117',
}

_THEME = DARK

# ── Custom CSS & helper functions ──────────────────────────────────

CSS_TEMPLATE = """\
* {{ -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }}
.stApp {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; }}
h1 {{ font-size: 1.75rem !important; font-weight: 700 !important; color: {text} !important; }}
h2 {{ font-size: 1.35rem !important; font-weight: 600 !important; color: {text} !important; padding-bottom: 0.4rem !important; }}
h3 {{ font-size: 1.1rem !important; font-weight: 600 !important; color: {text2} !important; }}

* {{ scrollbar-width: thin; scrollbar-color: {scrollbarThumb} {scrollbarTrack}; }}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: {scrollbarTrack}; border-radius: 4px; }}
::-webkit-scrollbar-thumb {{ background: {scrollbarThumb}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {scrollbarThumbHover}; }}

[data-testid="stSidebar"] {{ background-color: {sidebarBg}; border-right: 1px solid {border2}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

.card {{
    background: {bg2}; border: 1px solid {border}; border-radius: 8px;
    padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px {cardShadow};
}}
.card-header {{
    background: linear-gradient(135deg, {bg2} 0%, {headerBg} 100%);
    border-bottom: 1px solid {border}; border-radius: 8px 8px 0 0;
    padding: 16px 20px; margin: -20px -20px 16px -20px;
}}
.card-hallucination {{
    background: {bg2}; border: 1px solid {border}; border-left: 4px solid {danger};
    border-radius: 0 8px 8px 0; padding: 14px 16px; margin-bottom: 10px;
}}
.card-normal {{
    background: {bg2}; border: 1px solid {border}; border-left: 4px solid {success};
    border-radius: 0 8px 8px 0; padding: 14px 16px; margin-bottom: 10px;
}}
.card-uncertain {{
    background: {bg2}; border: 1px solid {border}; border-left: 4px solid {warning};
    border-radius: 0 8px 8px 0; padding: 14px 16px; margin-bottom: 10px;
}}

div[data-testid="stMetric"] {{
    background: {bg2}; border: 1px solid {border}; border-radius: 8px;
    padding: 16px 20px !important; box-shadow: 0 1px 3px {cardShadow};
}}
div[data-testid="stMetricValue"] {{ font-size: 1.8rem !important; font-weight: 700 !important; color: {primary} !important; }}
div[data-testid="stMetricLabel"] {{ color: {text3} !important; font-size: 0.8rem !important; font-weight: 500 !important; }}

.stButton > button {{
    border-radius: 6px !important; font-weight: 600 !important; font-size: 0.875rem !important; transition: all 0.15s ease !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {btnGradientStart} 0%, {btnGradientEnd} 100%) !important;
    border: none !important; box-shadow: 0 2px 6px {btnShadow} !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 4px 12px {btnHoverShadow} !important; transform: translateY(-1px);
}}
.stButton > button[kind="secondary"] {{
    background: {btnSecondaryBg} !important; border: 1px solid {border} !important; color: {text2} !important;
}}
.stButton > button[kind="secondary"]:hover {{ border-color: {primary} !important; color: {primary} !important; }}

[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
[data-testid="stTable"] {{ border-radius: 8px; overflow: hidden; border: 1px solid {border}; }}
[data-testid="stTable"] thead th {{
    background: {headerBg} !important; color: {text3} !important; font-weight: 600 !important;
    font-size: 0.8rem !important; padding: 10px 14px !important; border-bottom: 2px solid {border} !important;
}}
[data-testid="stTable"] tbody td {{
    padding: 10px 14px !important; border-bottom: 1px solid {border2} !important;
    color: {text2} !important; font-size: 0.875rem !important;
}}
[data-testid="stTable"] tbody tr:hover {{ background: {primaryHoverBg} !important; }}

[data-testid="stExpander"] details {{ border: 1px solid {border} !important; border-radius: 8px !important; background: {bg2} !important; }}
[data-testid="stExpander"] summary {{
    padding: 12px 16px !important; font-weight: 600 !important; color: {primary} !important; border-radius: 8px !important;
}}
[data-testid="stExpander"] summary:hover {{ background: {primaryHoverBg2} !important; }}
.streamlit-expanderContent {{ padding: 16px !important; border-top: 1px solid {border} !important; }}

[data-testid="stFileUploader"] section {{
    border: 2px dashed {border} !important; border-radius: 10px !important;
    background: {bg2} !important; padding: 2rem !important; transition: border-color 0.2s ease !important;
}}
[data-testid="stFileUploader"] section:hover {{ border-color: {primary} !important; background: {primaryHoverBg3} !important; }}

hr {{ border-color: {border2} !important; margin: 1.5rem 0 !important; }}
div[data-testid="stAlert"] {{ border-radius: 8px !important; border: 1px solid {border} !important; }}

[data-testid="stSelectbox"] > div, [data-testid="stTextInput"] > div {{ border-radius: 6px !important; }}

@keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.6; }} }}
@keyframes slideUp {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: translateY(0); }} }}

.badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
.badge-danger  {{ background: {dangerBg}; color: {danger}; }}
.badge-success {{ background: {successBg}; color: {success}; }}
.badge-warning {{ background: {warningBg}; color: {warning}; }}
.badge-info    {{ background: {primaryBg}; color: {primary}; }}"""


def inject_custom_css() -> None:
    """Inject custom CSS."""
    css = CSS_TEMPLATE.format(**_THEME)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def card_open(css_class: str = "card") -> None:
    """Open a styled card container."""
    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)


def card_close() -> None:
    """Close the card container."""
    st.markdown('</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, color: str = "#4F8CFF", subtitle: str = "") -> None:
    """Render a custom styled metric card using the current theme."""
    t = _THEME
    sub = f'<div style="font-size:0.8rem;color:{t["text4"]};margin-top:4px;">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div style="background:{t['bg2']};border:1px solid {t['border']};border-radius:8px;
                padding:18px 20px;text-align:center;margin-bottom:8px;
                box-shadow:0 1px 3px {t['cardShadow']};">
        <div style="font-size:0.75rem;color:{t['text3']};text-transform:uppercase;
                    letter-spacing:0.5px;margin-bottom:6px;">{label}</div>
        <div style="font-size:2rem;font-weight:700;color:{color};line-height:1.2;">{value}</div>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def theme_color(key: str) -> str:
    """Get a color value from the current theme."""
    return _THEME[key]


def main() -> None:
    inject_custom_css()

    st.title("客服回复幻觉检测系统")
    st.caption("智能检测客服回复中的政策编造、参数错误和能力越界等问题")
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    init_db()

    pages = {
        "批次管理": page_batch_history,
        "上传检测": page_upload,
        "结果浏览": page_results,
        "评估分析": page_evaluation,
        "误判分析": page_misclassification,
    }

    # Restore page from URL param on browser refresh
    if "nav_page" not in st.session_state:
        qp = st.query_params
        if qp.get("page") in pages:
            st.session_state.nav_page = qp["page"]

    # Handle programmatic navigation before widget instantiation
    if nav := st.session_state.pop("_nav_request", None):
        if nav in pages:
            st.session_state.nav_page = nav

    with st.sidebar:
        st.title("导航")
        page = st.radio("选择页面", list(pages.keys()), key="nav_page")

    # Persist current page to URL
    if st.query_params.get("page") != page:
        st.query_params["page"] = page

    pages[page]()


def page_batch_history() -> None:
    st.header("检测批次历史")

    conn = get_connection()
    try:
        batches = conn.execute(
            "SELECT * FROM detection_batches ORDER BY id DESC"
        ).fetchall()

        if not batches:
            st.info("暂无检测记录，请先上传数据进行检测。")
            return

        # ── KPI summary cards ──
        total_batches = len(batches)
        total_detections = sum(b["total_count"] for b in batches)
        total_hallu = sum(b["hallucination_count"] for b in batches)
        avg_rate = total_hallu / total_detections * 100 if total_detections > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("总批次数", str(total_batches), color=theme_color('primary'))
        with k2:
            metric_card("总检测数", f"{total_detections:,}", color=theme_color('purple'))
        with k3:
            metric_card("总幻觉数", f"{total_hallu:,}", color=theme_color('danger'))
        with k4:
            metric_card("平均幻觉率", f"{avg_rate:.1f}%", color=theme_color('warning'))

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        # ── Batch table with colored hallucination rate ──
        data = []
        for b in batches:
            data.append({
                "批次ID": b["id"],
                "文件名": b["filename"],
                "检测时间": b["created_at"],
                "总数": b["total_count"],
                "幻觉数": b["hallucination_count"],
                "幻觉率": f"{b['hallucination_count'] / b['total_count'] * 100:.1f}%"
                if b["total_count"] > 0 else "0%",
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        if "batch_id" not in st.session_state:
            st.session_state.batch_id = batches[0]["id"]

        card_open("card")
        selected_id = st.selectbox(
            "选择批次查看详情",
            [b["id"] for b in batches],
            format_func=lambda x: f"批次 #{x}",
        )
        st.session_state.batch_id = selected_id
        card_close()

        # Show results for selected batch
        _show_batch_results(conn, selected_id)

    finally:
        conn.close()


def _show_batch_results(conn, batch_id: int) -> None:
    """Display detection results for a given batch with filters."""
    st.divider()
    st.subheader(f"批次 #{batch_id} 检测详情")

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    # ── Filter card ──
    card_open("card")
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox(
            "幻觉类型",
            ["全部", "政策编造", "参数编造", "政策偏差", "能力越界",
             "优惠编造", "信息编造", "安全误导", "信息遗漏"],
            key=f"type_{batch_id}",
        )
    with col2:
        hallucination_filter = st.selectbox(
            "是否幻觉", ["全部", "是", "否"], key=f"hallu_{batch_id}",
        )
    with col3:
        search = st.text_input("搜索关键词", placeholder="输入回复内容关键词...", key=f"search_{batch_id}")
    card_close()

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    conditions = ["batch_id = ?"]
    params: list = [batch_id]

    if type_filter != "全部":
        conditions.append("output_type = ?")
        params.append(type_filter)
    if hallucination_filter == "是":
        conditions.append("is_hallucination = 1")
    elif hallucination_filter == "否":
        conditions.append("is_hallucination = 0")
    if search:
        conditions.append("(system_reply LIKE ? OR user_question LIKE ?)")
        search_param = f"%{search}%"
        params.extend([search_param, search_param])

    where = " AND ".join(conditions)
    rows = conn.execute(
        f"SELECT * FROM detection_results WHERE {where} ORDER BY id",
        params,
    ).fetchall()

    # ── Export buttons ──
    all_rows = conn.execute(
        "SELECT * FROM detection_results WHERE batch_id = ? ORDER BY id",
        [batch_id],
    ).fetchall()
    export_data = []
    for r in all_rows:
        export_data.append({
            "id": r["reply_id"],
            "user_question": r["user_question"],
            "system_reply": r["system_reply"],
            "knowledge_base": r["knowledge_base"],
            "is_hallucination": bool(r["is_hallucination"]) if r["is_hallucination"] is not None else None,
            "detection_layer": r["detection_layer"],
            "output_type": r["output_type"],
            "confidence": r["confidence"],
            "reason": r["reason"],
        })

    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 3])
    with col_dl1:
        st.download_button(
            "导出 JSON",
            json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"detection_results_batch_{batch_id}.json",
            mime="application/json",
            key=f"dl_json_{batch_id}",
        )
    with col_dl2:
        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "导出 CSV",
            csv,
            file_name=f"detection_results_batch_{batch_id}.csv",
            mime="text/csv",
            key=f"dl_csv_{batch_id}",
        )

    if not rows:
        st.info("没有匹配的结果。")
        return

    total = len(rows)
    hallu = sum(1 for r in rows if r["is_hallucination"])
    t = _THEME
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:16px 0 8px 0;">
        <span style="color:{t['text3']};font-size:0.85rem;">筛选结果</span>
        <span style="color:{t['text']};font-size:1.3rem;font-weight:700;">{total} 条</span>
        <span class="badge badge-danger">
            幻觉 {hallu} 条 ({hallu/total*100:.0f}%)
        </span>
    </div>
    """, unsafe_allow_html=True)

    for row in rows:
        is_hallu = row["is_hallucination"]
        if is_hallu is True:
            card_class = "card-hallucination"
            badge_class = "badge-danger"
            status_text = "幻觉"
        elif is_hallu is False:
            card_class = "card-normal"
            badge_class = "badge-success"
            status_text = "正常"
        else:
            card_class = "card-uncertain"
            badge_class = "badge-warning"
            status_text = "未判定"

        out_type = row["output_type"] or "-"
        confidence = row["confidence"] or "-"
        t = _THEME

        st.markdown(f"""
        <div class="{card_class}" style="animation:slideUp 0.25s ease;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <span style="color:{t['text']};font-weight:600;font-size:0.9rem;">{row['reply_id']}</span>
                <span class="badge {badge_class}">{status_text}</span>
                <span style="color:{t['text4']};font-size:0.8rem;">{out_type}</span>
                <span style="color:{t['text3']};font-size:0.8rem;margin-left:auto;">置信度：{confidence}</span>
            </div>
            <div style="color:{t['text3']};font-size:0.85rem;line-height:1.5;">
                {(row['reason'] or '')[:120]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("查看详情", key=f"detail_batch_{batch_id}_{row['id']}"):
            _show_detail(row)


def page_upload() -> None:
    st.header("上传数据并检测")

    if "_uploader_version" not in st.session_state:
        st.session_state._uploader_version = 0

    card_open("card")
    uploaded_file = st.file_uploader(
        "上传 replies.json 文件", type=["json"],
        key=f"fu_{st.session_state._uploader_version}",
    )
    card_close()

    if uploaded_file is not None:
        content = uploaded_file.read()
        try:
            data = json.loads(content.decode("utf-8"))
        except json.JSONDecodeError:
            st.error("JSON 格式无效，请检查文件。")
            return

        if not isinstance(data, list):
            st.error("JSON 必须是一个数组。")
            return

        t = _THEME
        st.markdown(f"""
        <div style="background:{t['successBg']};border:1px solid {t['success']}44;
                    border-radius:8px;padding:12px 16px;margin:8px 0;">
            <span style="color:{t['success']};font-weight:600;">成功加载 {len(data)} 条回复数据</span>
        </div>
        """, unsafe_allow_html=True)

        # Preview
        with st.expander("数据预览"):
            preview = []
            for item in data[:10]:
                preview.append({
                    "ID": item.get("id", "?"),
                    "用户问题": item["user_question"][:50] + "...",
                    "系统回复": item["system_reply"][:80] + "...",
                })
            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        if st.button("开始检测", type="primary", use_container_width=True):
            st.session_state._detecting = True
            st.session_state._detection_error = None
            st.session_state._detection_result = None
            st.session_state._detection_file = (uploaded_file.name, data)
            st.rerun()

        # ── Detecting state ─────────────────────────────────────────
        if st.session_state.get("_detecting"):
            file_name, data_content = st.session_state._detection_file
            total = len(data_content)

            # Start background thread on first entry
            if "_detection_thread" not in st.session_state or st.session_state._detection_thread is None:
                result_holder: dict = {}
                st.session_state._result_holder = result_holder

                def _run(holder: dict) -> None:
                    payload = json.dumps(data_content, ensure_ascii=False)
                    files = {"file": (file_name, payload.encode("utf-8"), "application/json")}
                    try:
                        resp = requests.post(f"{API_BASE}/api/detect/upload", files=files, timeout=120)
                        if resp.status_code == 200:
                            holder["result"] = resp.json()
                        else:
                            holder["error"] = f"API 错误: {resp.status_code}"
                    except Exception as e:
                        holder["error"] = f"无法连接 API 服务: {e}"

                st.session_state._detection_thread = threading.Thread(
                    target=_run, args=(result_holder,), daemon=True,
                )
                st.session_state._detection_start = time.time()
                st.session_state._detection_thread.start()
                st.rerun()

            thread = st.session_state._detection_thread

            if thread.is_alive():
                elapsed = int(time.time() - st.session_state._detection_start)
                estimated_total = max(total * 3, 1)
                progress_pct = min(elapsed / estimated_total * 100, 95)

                t = _THEME
                st.markdown(f"""
                <div style="background:{t['bg2']};border:1px solid {t['border']};border-radius:10px;
                            padding:28px 24px;text-align:center;margin:16px 0;">
                    <div style="width:36px;height:36px;border:3px solid {t['border']};
                                border-top-color:{t['primary']};border-radius:50%;
                                animation:spin 1s linear infinite;margin:0 auto 16px auto;"></div>
                    <p style="color:{t['text2']};font-size:1rem;margin:8px 0;">
                        正在检测 <strong>{total}</strong> 条回复
                    </p>
                    <p style="color:{t['text3']};font-size:0.85rem;">已用时 {elapsed}s</p>
                    <div style="background:{t['border2']};border-radius:6px;height:8px;margin-top:16px;overflow:hidden;">
                        <div style="background:linear-gradient(90deg,{t['primary']},{t['purple']});
                                    width:{progress_pct:.0f}%;height:100%;border-radius:6px;
                                    transition:width 0.5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("取消检测", use_container_width=True):
                        st.session_state._detecting = False
                        st.session_state._detection_file = None
                        st.session_state._detection_thread = None
                        st.session_state._result_holder = None
                        st.session_state._uploader_version += 1
                        st.rerun()
                time.sleep(2)
                st.rerun()
            else:
                # Detection finished — read results from plain dict (thread-safe)
                st.session_state._detection_thread = None
                st.session_state._detecting = False

                holder = st.session_state.pop("_result_holder", {})
                error = holder.get("error")
                result = holder.get("result")

                if error:
                    st.session_state._detection_error = error
                elif result:
                    st.session_state._detection_result = result
                st.rerun()

        # ── Show detection result outside the detecting block ──────────
        if error := st.session_state.pop("_detection_error", None):
            st.error(error)
            st.info("请确保 FastAPI 服务已启动")

        if result := st.session_state.get("_detection_result"):
            st.toast("本次检测已完成", icon="✅")
            t = _THEME
            st.markdown(f"""
            <div style="background:{t['successBg']};border:1px solid {t['success']}44;
                        border-radius:10px;padding:20px 24px;margin:12px 0;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:1.5rem;">✅</span>
                    <span style="color:{t['text']};font-size:1rem;">
                        检测完成！共 <strong>{result['total']}</strong> 条，
                        发现 <strong style="color:{t['danger']};">{result['hallucination_count']}</strong> 条幻觉
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.batch_id = result.get("batch_id")
            st.session_state.latest_results = result.get("results", [])
            if st.button("查看本次检测结果", use_container_width=True, key="goto_results"):
                del st.session_state["_detection_result"]
                st.session_state._nav_request = "结果浏览"
                st.rerun()


def page_results() -> None:
    st.header("检测结果浏览")

    conn = get_connection()
    try:
        batch_id = st.session_state.get("batch_id")
        if not batch_id:
            batch = conn.execute(
                "SELECT id FROM detection_batches ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if batch:
                batch_id = batch["id"]
                st.session_state.batch_id = batch_id
            else:
                st.info("暂无检测结果。请先上传数据进行检测。")
                return

        _show_batch_results(conn, batch_id)

    finally:
        conn.close()


def _show_detail(row) -> None:
    """Show full detail of a detection result."""
    with st.expander("", expanded=True):
        st.markdown(f"**Reply ID:** {row['reply_id']}")
        st.markdown(f"**用户问题:** {row['user_question']}")
        st.markdown(f"**系统回复:** {row['system_reply']}")
        st.markdown(f"**知识库:** {row['knowledge_base']}")
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        is_hallu = row["is_hallucination"]
        col1.metric("是否幻觉", "是" if is_hallu else "否" if is_hallu is not None else "未判定")
        col2.metric("检测层", row["detection_layer"] or "-")
        col3.metric("输出类型", row["output_type"] or "-")
        col4.metric("置信度", row["confidence"] or "-")
        st.info(f"**判定依据:** {row['reason']}")


def page_evaluation() -> None:
    st.header("评估指标")

    card_open("card")
    t = _THEME
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<p style="color:{t["text3"]};font-size:0.85rem;margin-bottom:4px;">Ground Truth（人工标注）</p>', unsafe_allow_html=True)
        gt_file = st.file_uploader("上传 ground_truth.json", type=["json"], key="gt_upload", label_visibility="collapsed")
    with col2:
        st.markdown(f'<p style="color:{t["text3"]};font-size:0.85rem;margin-bottom:4px;">检测结果（系统输出）</p>', unsafe_allow_html=True)
        results_file = st.file_uploader("上传 detection_results.json", type=["json"], key="results_upload", label_visibility="collapsed")
    card_close()

    if gt_file and results_file:
        try:
            gt_data = json.loads(gt_file.read().decode("utf-8"))
            results_data = json.loads(results_file.read().decode("utf-8"))
        except json.JSONDecodeError as e:
            st.error(f"JSON 解析错误: {e}")
            return

        try:
            metrics = compute_metrics(gt_data, results_data)
        except (ValueError, KeyError) as e:
            st.error(f"文件格式错误: {e}")
            return

        st.divider()

        # ── Core metrics: custom colored cards ──
        t = _THEME
        st.markdown(f'<p style="color:{t["text3"]};font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">核心指标</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("准确率", f"{metrics.accuracy:.1%}", color=t['primary'], subtitle="Accuracy")
        with c2:
            metric_card("精确率", f"{metrics.precision:.1%}", color=t['success'], subtitle="Precision")
        with c3:
            metric_card("召回率", f"{metrics.recall:.1%}", color=t['warning'], subtitle="Recall")
        with c4:
            metric_card("F1 分数", f"{metrics.f1:.1%}", color=t['purple'], subtitle="F1 Score")

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        # ── Confusion matrix layout ──
        st.markdown(f'<p style="color:{t["text3"]};font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">混淆矩阵</p>', unsafe_allow_html=True)
        tl, tr = st.columns(2)
        bl, br = st.columns(2)
        with tl:
            metric_card("真阳性 TP", str(metrics.true_positives), color=t['success'], subtitle="正确检出幻觉")
        with tr:
            metric_card("误报 FP", str(metrics.false_positives), color=t['danger'], subtitle="误判为幻觉")
        with bl:
            metric_card("漏检 FN", str(metrics.false_negatives), color=t['warning'], subtitle="遗漏的幻觉")
        with br:
            metric_card("真阴性 TN", str(metrics.true_negatives), color=t['primary'], subtitle="正确排除")

        # ── Misclassification tables ──
        if metrics.false_negatives > 0:
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="color:{t['danger']};font-weight:600;">漏检</span>
                <span class="badge badge-danger">{len(metrics.false_negative_cases)} 条</span>
            </div>
            """, unsafe_allow_html=True)
            fn_data = []
            for c in metrics.false_negative_cases:
                fn_data.append({
                    "ID": c["id"],
                    "实际类型": c["actual"],
                    "标注详情": c["ground_truth_detail"][:150],
                    "检测判定": c["detection_reason"][:100],
                })
            st.dataframe(pd.DataFrame(fn_data), use_container_width=True, hide_index=True)

        if metrics.false_positives > 0:
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="color:{t['warning']};font-weight:600;">误报</span>
                <span class="badge badge-warning">{len(metrics.false_positive_cases)} 条</span>
            </div>
            """, unsafe_allow_html=True)
            fp_data = []
            for c in metrics.false_positive_cases:
                fp_data.append({
                    "ID": c["id"],
                    "检测类型": c["output_type"],
                    "标注详情": c["ground_truth_detail"][:150],
                    "检测判定": c["detection_reason"][:100],
                })
            st.dataframe(pd.DataFrame(fp_data), use_container_width=True, hide_index=True)


def page_misclassification() -> None:
    st.header("误判分析")

    patterns = [
        {
            'title': '措辞差异误报 (h16)',
            'icon': '🔄',
            'color': '#D29922',
            'items': [
                ('现象', '知识库和回复表述方向不同但语义一致，导致被误判为幻觉'),
                ('案例', 'KB 说“可能有色差”，回复说“颜色基本准确，有轻微色差” → 语义等价但措辞相反'),
                ('原因', '规则引擎对方向性措辞敏感，LLM 通常能正确判断'),
            ],
        },
        {
            'title': '信息遗漏漏检 (h20)',
            'icon': '📋',
            'color': '#F85149',
            'items': [
                ('现象', '回复说的不算错，但遗漏了关键限制信息，导致漏检'),
                ('案例', 'KB 说“30% 用户反馈偏大半码”，回复说“尺码标准不偏”'),
                ('原因', '软性错误规则引擎难以捕获，LLM 需要理解不完整信息即误导'),
            ],
        },
        {
            'title': '部分正确的混合回复 (h04)',
            'icon': '⚡',
            'color': '#D29922',
            'items': [
                ('现象', '回复中既有正确信息也有错误信息，容易漏判'),
                ('案例', '“支持电子发票” (✓) + “支持纸质发票” (✗)，混合在一起'),
                ('原因', '实体对齐只能捕获多了一个实体，需要 LLM 做逐句判定'),
            ],
        },
        {
            'title': '安全相关漏检 (h13)',
            'icon': '⚠️',
            'color': '#F85149',
            'items': [
                ('现象', '需要专业知识才能识别风险，容易被忽略'),
                ('案例', '“视黄醇衍生物对孕妇的影响”需要医学背景知识'),
                ('原因', '依赖 LLM 的预训练知识，特殊领域可能存在知识盲区'),
            ],
        },
    ]

    t = _THEME
    for p in patterns:
        items_html = ""
        for label, text in p["items"]:
            items_html += f"""<div style="margin-bottom:8px;">
<span style="color:{t['text4']};font-size:0.8rem;font-weight:600;">{label}</span>
<span style="color:{t['text2']};font-size:0.875rem;">{text}</span>
</div>"""

        st.markdown(textwrap.dedent(f"""
        <div class="card" style="border-left:4px solid {p['color']};margin-bottom:16px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <span style="font-size:1.2rem;">{p['icon']}</span>
                <span style="font-weight:600;color:#E6EDF3;font-size:1rem;">{p['title']}</span>
            </div>
            {items_html}
        </div>
        """), unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    # Show misclassification data if available
    conn = get_connection()
    try:
        evals = conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if evals:
            st.divider()
            st.subheader("最近评估详情")

            ev1, ev2, ev3, ev4 = st.columns(4)
            with ev1:
                metric_card("准确率", f"{evals['accuracy']:.1%}", color=t['primary'], subtitle="Accuracy")
            with ev2:
                metric_card("精确率", f"{evals['precision']:.1%}", color=t['success'], subtitle="Precision")
            with ev3:
                metric_card("召回率", f"{evals['recall']:.1%}", color=t['warning'], subtitle="Recall")
            with ev4:
                metric_card("F1 分数", f"{evals['f1']:.1%}", color=t['purple'], subtitle="F1")

            details = json.loads(evals["details"])
            if details.get("false_negatives") or details.get("false_positives"):
                st.markdown(f"""
                <div style="display:flex;gap:16px;margin-top:12px;">
                    <span class="badge badge-danger">漏检 {len(details.get('false_negatives', []))} 条</span>
                    <span class="badge badge-warning">误报 {len(details.get('false_positives', []))} 条</span>
                </div>
                """, unsafe_allow_html=True)
    except Exception:
        pass
    finally:
        conn.close()


if __name__ == "__main__":
    main()
