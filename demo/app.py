"""Shopping Agent portfolio UI.

Sections deliberately distinguish:
- local catalog browsing (not the Agent retriever),
- live API calls (LLM quota required),
- recorded trace replay (no LLM call),
- fixed Day19 benchmark snapshots (not a new run).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import streamlit as st
import requests

from api_client import get_answer, get_events, health_check, send_chat

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
FIXTURES = BASE / "fixtures"

st.set_page_config(page_title="Shopping Agent | Portfolio Demo", page_icon="🛒", layout="wide")
st.title("🛒 Shopping Agent")
st.caption("Multi-Agent Research & Recommendation · Portfolio Demo")
st.info("展示模式清晰区分：本地数据浏览、实时 Agent、历史记录回放、历史 Benchmark。它们不是同一个运行流程。")

with st.sidebar:
    st.header("连接设置")
    api_url = st.text_input(
        "FastAPI URL",
        value=os.getenv(
            "DEMO_API_URL",
            "http://127.0.0.1:8000",
        ),
    )
    if st.button("检查 /health"):
        ok, msg = health_check(api_url)
        (st.success if ok else st.error)(msg)
    st.divider()
    st.write("**本地目录**")
    catalog_path_text = st.text_input("JSON 文件路径（相对项目根目录）", "data/products_real.json")
    st.caption("不会修改 Catalog；不会自动触发 Groq 调用。")
    st.divider()
    st.caption("警告：不要在公开视频或截图中暴露密钥、Cookie、私人对话或原始敏感日志。")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def load_catalog(path: Path) -> tuple[list[dict], str | None]:
    if not path.is_file():
        return [], f"文件不存在：{path}"
    data, error = load_json(path)
    if error:
        return [], error
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)], None
    if isinstance(data, dict):
        for key in ("products", "items", "catalog"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)], None
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values()), None
    return [], "无法识别目录结构；请检查是否为 JSON 数组或 {'products': [...]}。"


# STREAMLIT_TRACE_TIMELINE_V1
def render_trace(events: list[dict]):
    """Display existing events in append order, not a fabricated wall-clock Gantt."""
    import math

    if not events:
        st.info("当前记录没有逐节点 Trace events。历史评测摘要不会自动变成节点事件。")
        return

    valid_events = [item for item in events if isinstance(item, dict)]
    if not valid_events:
        st.warning("事件列表没有可展示的对象记录。")
        return

    st.caption(
        f"共有 {len(valid_events)} 条已记录事件。以下按保存顺序展示；"
        "并行 Candidate Workers 的执行区间可能重叠。"
        "每条耗时条只表示该事件自身耗时占本次最长事件的比例，不表示真实开始时间。"
    )
    st.markdown("#### Agent Execution Timeline")

    title_map = {
        "supervisor": "🧭 Supervisor",
        "research": "🔎 Research",
        "candidate_research": "🧪 Candidate Worker",
        "recommendation": "📝 Recommendation",
    }

    def duration_ms(event):
        value = event.get("latency_ms", event.get("duration_ms"))
        try:
            number = float(value)
        except (ValueError, TypeError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    durations = [duration_ms(item) for item in valid_events]
    max_duration = max((v for v in durations if v is not None), default=0.0)
    rows = []

    for index, (event, duration) in enumerate(zip(valid_events, durations), 1):
        kind = str(event.get("node") or event.get("agent") or event.get("event_type") or "unknown")
        kind_key = kind.lower()
        title = title_map.get(kind_key, kind)
        action = str(event.get("name") or event.get("tool") or event.get("tool_name") or "—")
        step = event.get("step", "—")
        raw_status = event.get("success")
        if isinstance(raw_status, bool):
            status = "成功" if raw_status else "失败"
        else:
            status_value = event.get("status")
            status = str(status_value) if status_value is not None else "未记录"

        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.write(f"**{index:02d}. {title}**")
                st.caption(f"Step: {step} · Action: {action}")
            with right:
                st.write(status)
                st.caption(f"{duration:,.1f} ms" if duration is not None else "耗时未记录")
            if duration is not None and max_duration > 0:
                st.progress(float(duration / max_duration))

        rows.append({
            "#": index,
            "Agent/Node": kind,
            "Event": str(event.get("event_type", event.get("type", "—"))),
            "Name / Action": action,
            "Tool": str(event.get("tool", event.get("tool_name", "—"))),
            "Latency (ms)": duration,
            "Status": status,
        })

    st.markdown("#### Event Details")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("查看原始 Trace JSON（分享前检查隐私）"):
        st.json(events)


catalog_tab, live_tab, replay_tab, eval_tab = st.tabs(
    ["📦 离线商品目录", "💬 Live Agent", "🧭 Trace Replay", "📊 Retrieval Evaluation"]
)

with catalog_tab:
    st.subheader("本地目录浏览（无 LLM 调用）")
    st.caption("这里只是 JSON 字段过滤和关键词包含匹配，**不代表** BM25/E5/RRF/CE 或 LangGraph 的执行结果。")
    catalog_path = Path(catalog_path_text).expanduser()
    if not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    products, err = load_catalog(catalog_path)
    if err:
        st.warning(err)
        st.code(f"ls -lh {catalog_path_text}", language="bash")
    else:
        st.metric("当前加载商品数", len(products))
        categories = sorted({str(p.get("category", "")) for p in products if p.get("category")})
        col1, col2 = st.columns([2, 1])
        with col1:
            keyword = st.text_input("商品名称 / 描述关键词（字符串包含）", "")
        with col2:
            category = st.selectbox("商品类别", ["全部"] + categories)
        matched = []
        for product in products:
            if category != "全部" and str(product.get("category")) != category:
                continue
            combined = " ".join(str(product.get(k, "")) for k in ("name", "title", "description", "brand", "tags"))
            if keyword and keyword.casefold() not in combined.casefold():
                continue
            matched.append(product)
        st.caption(f"按上述本地条件匹配 {len(matched)} 件；表格最多展示前 30 件。")
        st.dataframe([
            {
                "ID": p.get("id", p.get("parent_asin", "")),
                "商品名": p.get("name", p.get("title", "")),
                "类别": p.get("category", ""),
                "品牌": p.get("brand", ""),
                "价格": p.get("price", ""),
                "评分": p.get("rating", ""),
            }
            for p in matched[:30]
        ], use_container_width=True, hide_index=True)

with live_tab:
    st.subheader("连接真实 FastAPI `/chat`")
    st.warning("Live Mode 会触发真实 Agent 和可能的付费 / 限额 LLM 调用。额度不足时不要点击发送。")
    st.caption("真实请求字段需要先按 `http://127.0.0.1:8000/docs` 或 `/openapi.json` 检查；默认字段仅为示例。")
    if "demo_session_id" not in st.session_state:
        st.session_state.demo_session_id = str(uuid.uuid4())
    if "live_messages" not in st.session_state:
        st.session_state.live_messages = []
    opts1, opts2, opts3 = st.columns(3)
    with opts1:
        msg_field = st.text_input("消息字段名", "message")
    with opts2:
        sess_field = st.text_input("会话字段名", "session_id")
    with opts3:
        include_session = st.checkbox("发送 session_id", True)
    extras_text = st.text_area("额外 Request JSON 字段（可选）", value="{}", height=75)
    if st.button("重置前端会话"):
        st.session_state.demo_session_id = str(uuid.uuid4())
        st.session_state.live_messages = []
        st.rerun()
    st.caption(f"前端 session_id: `{st.session_state.demo_session_id}`")
    for role, content in st.session_state.live_messages:
        with st.chat_message(role):
            st.write(content)
    prompt = st.chat_input("输入一个真实商品需求（会消耗 API 额度）")
    if prompt:
        try:
            extra = json.loads(extras_text)
            if not isinstance(extra, dict):
                raise ValueError("额外 JSON 必须为对象 {}")
            payload = dict(extra)
            payload[msg_field] = prompt
            if include_session:
                payload[sess_field] = st.session_state.demo_session_id
            st.session_state.live_messages.append(("user", prompt))
            with st.chat_message("user"):
                st.write(prompt)
            with st.status("请求真实 Agent ...", expanded=True) as status:
                st.write(f"POST {api_url.rstrip('/')}/chat")
                data = send_chat(api_url, payload)
                status.update(label="请求完成", state="complete")
            answer = get_answer(data)
            with st.chat_message("assistant"):
                if answer:
                    st.write(answer)
                    st.session_state.live_messages.append(("assistant", answer))
                else:
                    st.info("未发现常见回答字段，显示原始响应；需要调整 api_client.get_answer。")
                    st.json(data)
            with st.expander("本次请求返回的 Trace（如有）"):
                render_trace(get_events(data))
            with st.expander("查看返回 JSON（请注意隐私）"):
                st.json(data)
        except requests.HTTPError as exc:
            st.error(f"HTTP 请求失败：{exc}")
            if exc.response is not None:
                st.code(exc.response.text[:2500], language="json")
            st.info("HTTP 422 通常表示请求字段不匹配；先查看 FastAPI `/docs`，不要修改 Agent 核心逻辑。")
        except Exception as exc:
            st.error(f"请求未完成：{exc}")

with replay_tab:
    st.subheader("历史 Agent Trace 回放（不访问 API）")
    fixture_paths = sorted(FIXTURES.glob("*.json"))
    if not fixture_paths:
        st.warning("没有回放文件；请将真实记录按 README 规范放到 demo/fixtures/。")
    else:
        chosen = st.selectbox("选择记录", fixture_paths, format_func=lambda p: p.name)
        fixture, error = load_json(chosen)
        if error or not isinstance(fixture, dict):
            st.error(f"无法解析 JSON：{error}")
        else:
            source = fixture.get("source", "unspecified")
            if source == "synthetic_ui_example":
                st.warning("本文件是虚构的 UI 样例，仅测试布局；不是项目真实运行证据，不可用于简历。")
            elif source == "recorded_real_run":
                st.success("文件标记为真实运行回放。请确认数据来源并清除隐私信息。")
            elif source == "recorded_eval_summary":
                st.info("从 V4b 的历史 Agent 评测报告导入：这是真实运行摘要，不是逐节点 Trace。")
                st.caption(f"来源报告：{fixture.get('source_file', '未标注')}")
            else:
                st.info(f"来源未验证：{source}。请核实后再用于面试。")
            st.write(f"**场景：** {fixture.get('scenario', '未填写')}")
            st.write(f"**用户需求：** {fixture.get('query', '未填写')}")
            st.write(f"**结果：** {fixture.get('answer', '未填写')}")
            if source == "recorded_eval_summary":
                st.markdown("**历史评测中实际记录的路由、工具与端到端耗时**")
                st.dataframe([{
                    "实际路由": fixture.get("actual_route", "—"),
                    "实际工具": fixture.get("actual_tool", "—"),
                    "端到端耗时 (ms)": fixture.get("latency_ms"),
                    "Task Success": fixture.get("task_success"),
                }], use_container_width=True, hide_index=True)
                st.caption(fixture.get("limitations", "未包含逐节点 Trace"))
            else:
                events = get_events(fixture)
                render_trace(events)

with eval_tab:
    st.subheader("Day 19 WANDS — 已完成实验结果")
    st.caption("此处是根据已上传项目迁移文档录入的历史结果快照，不会启动新评测；投递前请与原始 reports 交叉核对。")
    c1, c2, c3 = st.columns(3)
    c1.metric("Corpus", "42,994")
    c2.metric("Held-out queries", "384")
    c3.metric("RRF + CE@20 nDCG@10", "0.7662")
    baseline = [
        {"method": "BM25", "nDCG@10": 0.6845, "Exact MRR@10": 0.7017},
        {"method": "Dense", "nDCG@10": 0.7130, "Exact MRR@10": 0.7135},
        {"method": "RRF", "nDCG@10": 0.7388, "Exact MRR@10": 0.7240},
        {"method": "Dense + CE@20", "nDCG@10": 0.7482, "Exact MRR@10": 0.7609},
        {"method": "RRF + CE@20", "nDCG@10": 0.7662, "Exact MRR@10": 0.7598},
    ]
    st.bar_chart(baseline, x="method", y="nDCG@10", y_label="nDCG@10")
    st.dataframe(baseline, hide_index=True, use_container_width=True)
    st.markdown("**Dev ablation — CrossEncoder depth selection**")
    ablation = [
        {"configuration": "RRF + CE@10", "nDCG@10": 0.7479, "CE latency (ms)": 411},
        {"configuration": "RRF + CE@20", "nDCG@10": 0.7695, "CE latency (ms)": 787},
        {"configuration": "RRF + CE@50", "nDCG@10": 0.7648, "CE latency (ms)": 1911},
    ]
    st.dataframe(ablation, hide_index=True, use_container_width=True)
    st.caption("CE@20 相比 CE@50 的 CE 阶段平均延迟降低约 58.8%。不是 Agent 端到端延迟。WANDS relevance judgments 为 pooled / non-exhaustive。")
    st.warning("Day 20 的首次 Agent held-out 因 Groq TPD 配额耗尽而无效；本页面不把 25% Task Success 当作成绩。")
