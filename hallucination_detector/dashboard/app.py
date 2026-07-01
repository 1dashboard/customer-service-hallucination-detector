"""Streamlit dashboard for hallucination detection results."""

from __future__ import annotations

import json
import sys
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


def main() -> None:
    st.title("🔍 客服回复幻觉检测系统")

    init_db()

    pages = {
        "批次管理": page_batch_history,
        "上传检测": page_upload,
        "结果浏览": page_results,
        "评估分析": page_evaluation,
        "误判分析": page_misclassification,
    }

    with st.sidebar:
        st.title("导航")
        page = st.radio("选择页面", list(pages.keys()))

    pages[page]()


def page_batch_history() -> None:
    st.header("📋 检测批次历史")

    conn = get_connection()
    try:
        batches = conn.execute(
            "SELECT * FROM detection_batches ORDER BY id DESC"
        ).fetchall()

        if not batches:
            st.info("暂无检测记录，请先上传数据进行检测。")
            return

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

        if "batch_id" not in st.session_state:
            st.session_state.batch_id = batches[0]["id"]

        selected_id = st.selectbox(
            "选择批次查看详情",
            [b["id"] for b in batches],
            format_func=lambda x: f"批次 #{x}",
        )
        st.session_state.batch_id = selected_id

    finally:
        conn.close()


def page_upload() -> None:
    st.header("📤 上传数据并检测")

    if "_uploader_key" not in st.session_state:
        st.session_state._uploader_key = 0

    uploaded_file = st.file_uploader(
        "上传 replies.json 文件", type=["json"],
        key=f"file_uploader_{st.session_state._uploader_key}",
    )

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

        st.success(f"成功加载 {len(data)} 条回复数据")

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

        if st.button("🚀 开始检测", type="primary", use_container_width=True):
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
                col1, col2 = st.columns([3, 1])
                with col1:
                    elapsed = int(time.time() - st.session_state._detection_start)
                    st.info(f"⏳ 正在检测 {total} 条回复，已用时 {elapsed}s...")
                with col2:
                    if st.button("⏹ 取消检测", use_container_width=True):
                        st.session_state._detecting = False
                        st.session_state._detection_file = None
                        st.session_state._detection_thread = None
                        st.session_state._result_holder = None
                        st.session_state._uploader_key += 1
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
                    st.error(error)
                    st.info("请确保 FastAPI 服务已启动")
                elif result:
                    st.balloons()
                    st.toast("✅ 本次检测已完成", icon="✅")
                    st.success(
                        f"检测完成！共 {result['total']} 条，"
                        f"发现 {result['hallucination_count']} 条幻觉"
                    )
                    st.session_state.batch_id = result.get("batch_id")
                    st.session_state.latest_results = result.get("results", [])


def page_results() -> None:
    st.header("📊 检测结果浏览")

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

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            type_filter = st.selectbox(
                "幻觉类型",
                ["全部", "政策编造", "参数编造", "政策偏差", "能力越界",
                 "优惠编造", "信息编造", "安全误导", "信息遗漏"],
            )
        with col2:
            hallucination_filter = st.selectbox(
                "是否幻觉", ["全部", "是", "否"]
            )
        with col3:
            search = st.text_input("搜索关键词", placeholder="输入回复内容关键词...")

        # Build query
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

        if not rows:
            st.info("没有匹配的结果。")
            return

        # Stats
        total = len(rows)
        hallu = sum(1 for r in rows if r["is_hallucination"])
        st.metric("筛选结果", f"{total} 条", f"幻觉 {hallu} 条 ({hallu/total*100:.0f}%)" if total else "")

        # Table
        for row in rows:
            is_hallu = row["is_hallucination"]
            bg_color = "#ffcccc" if is_hallu else ("#ccffcc" if is_hallu is not None else "#f0f0f0")
            status_text = "⚠️ 幻觉" if is_hallu else ("✅ 正常" if is_hallu is not None else "❓ 未判定")

            with st.container():
                cols = st.columns([1, 2, 6, 2, 2, 1])
                cols[0].markdown(f"**{row['reply_id']}**")
                cols[1].markdown(f"<span style='color:{'red' if is_hallu else 'green'}'>{status_text}</span>",
                                 unsafe_allow_html=True)
                cols[2].markdown(row["reason"][:100])
                out_type = row["output_type"] or "-"
                confidence = row["confidence"] or "-"
                cols[3].markdown(f"`{out_type}`")
                cols[4].markdown(f"`{confidence}`")

                if cols[5].button("详情", key=f"detail_{row['id']}"):
                    _show_detail(row)

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
        col1, col2, col3 = st.columns(3)
        is_hallu = row["is_hallucination"]
        col1.metric("是否幻觉", "是" if is_hallu else "否" if is_hallu is not None else "未判定")
        col2.metric("检测层", row["detection_layer"] or "-")
        col3.metric("输出类型", row["output_type"] or "-")
        st.info(f"**判定依据:** {row['reason']}")


def page_evaluation() -> None:
    st.header("📈 评估指标")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ground Truth")
        gt_file = st.file_uploader("上传 ground_truth.json", type=["json"], key="gt_upload")
    with col2:
        st.subheader("检测结果")
        results_file = st.file_uploader("上传 detection_results.json", type=["json"], key="results_upload")

    if gt_file and results_file:
        try:
            gt_data = json.loads(gt_file.read().decode("utf-8"))
            results_data = json.loads(results_file.read().decode("utf-8"))
        except json.JSONDecodeError as e:
            st.error(f"JSON 解析错误: {e}")
            return

        metrics = compute_metrics(gt_data, results_data)

        st.divider()
        st.subheader("核心指标")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics.accuracy:.1%}")
        col2.metric("Precision", f"{metrics.precision:.1%}")
        col3.metric("Recall", f"{metrics.recall:.1%}")
        col4.metric("F1 Score", f"{metrics.f1:.1%}")

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("True Positives", metrics.true_positives)
        col2.metric("True Negatives", metrics.true_negatives)
        col3, col4 = st.columns(2)
        col3.metric("False Positives (误报)", metrics.false_positives)
        col4.metric("False Negatives (漏检)", metrics.false_negatives)

        # Misclassification tables
        if metrics.false_negatives > 0:
            st.subheader(f"🔴 漏检 ({len(metrics.false_negative_cases)} 条)")
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
            st.subheader(f"🟡 误报 ({len(metrics.false_positive_cases)} 条)")
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
    st.header("🔬 误判分析")

    st.markdown("""
    ### 常见误判模式

    **1. 措辞差异误报 (h16)**
    - 现象: 知识库和回复表述方向不同但语义一致
    - 案例: KB说"可能有色差"，回复说"颜色基本准确，有轻微色差" → 语义等价但措辞相反
    - 原因: 规则引擎对"方向性措辞"敏感，LLM 通常能正确判断

    **2. 信息遗漏漏检 (h20)**
    - 现象: 回复说的不算错，但遗漏了关键限制信息
    - 案例: KB说"30%用户反馈偏大半码"，回复说"尺码标准不偏"
    - 原因: 这类"软性错误"规则引擎难以捕获，LLM 需要理解"不完整信息 = 误导"

    **3. 部分正确的混合回复 (h04)**
    - 现象: 回复中既有正确信息也有错误信息
    - 案例: "支持电子发票"(✓) + "支持纸质发票"(✗)
    - 原因: 实体对齐只能捕获"多了一个实体"，需要 LLM 做逐句判定

    **4. 安全相关漏检 (h13)**
    - 现象: 需要专业知识才能识别风险
    - 案例: "视黄醇衍生物对孕妇的影响"需要医学知识
    - 原因: 依赖 LLM 的预训练知识，可能存在知识盲区
    """)

    # Show misclassification data if available
    conn = get_connection()
    try:
        evals = conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if evals:
            st.subheader("最近评估详情")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{evals['accuracy']:.1%}")
            col2.metric("Precision", f"{evals['precision']:.1%}")
            col3.metric("Recall", f"{evals['recall']:.1%}")
            col4.metric("F1", f"{evals['f1']:.1%}")

            details = json.loads(evals["details"])
            if details.get("false_negatives"):
                st.markdown(f"**漏检:** {len(details['false_negatives'])} 条")
            if details.get("false_positives"):
                st.markdown(f"**误报:** {len(details['false_positives'])} 条")
    except Exception:
        pass
    finally:
        conn.close()


def page_export() -> None:
    st.header("💾 导出结果")

    conn = get_connection()
    try:
        batch_id = st.session_state.get("batch_id")
        if not batch_id:
            st.info("请先选择批次。")
            return

        rows = conn.execute(
            "SELECT * FROM detection_results WHERE batch_id = ?",
            [batch_id],
        ).fetchall()

        results = []
        for r in rows:
            results.append({
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

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 导出 JSON",
                json.dumps(results, ensure_ascii=False, indent=2),
                file_name="detection_results.json",
                mime="application/json",
            )
        with col2:
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 导出 CSV",
                csv,
                file_name="detection_results.csv",
                mime="text/csv",
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
