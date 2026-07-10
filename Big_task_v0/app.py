from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from config import settings
from models.schemas import AudiencePlan, ProjectResult, PublisherProfile
from services.orchestrator import CommentLabOrchestrator


ROOT = Path(__file__).resolve().parent


st.set_page_config(page_title="CommentLab", page_icon="CL", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.4rem;}
    h1, h2, h3 {letter-spacing: 0;}
    div[data-testid="stMetric"] {border-left: 3px solid #e6543b; padding-left: 0.8rem;}
    .comment-meta {color:#68717d; font-size:0.82rem; margin-bottom:0.2rem;}
    .comment-text {font-size:0.98rem; line-height:1.65;}
    .disclaimer {border-left:3px solid #d99b24; padding:0.55rem 0.8rem; background:#fff8e7; color:#514527;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_orchestrator() -> CommentLabOrchestrator:
    return CommentLabOrchestrator(settings)


@st.cache_data
def load_demo_cases() -> list[dict]:
    return json.loads((ROOT / "data" / "demo_cases.json").read_text(encoding="utf-8"))


def apply_demo_case() -> None:
    selected = st.session_state.get("case_selector", "自定义")
    case = next((item for item in load_demo_cases() if item["name"] == selected), None)
    if case is None:
        return
    st.session_state.input_post = case["post_text"]
    st.session_state.input_identity = case["identity"]
    st.session_state.input_domain = case["domain"]
    st.session_state.input_follower_scale = case["follower_scale"]
    st.session_state.input_style = case["style"]
    st.session_state.input_audience_relationship = case["audience_relationship"]


def reset_flow() -> None:
    for key in ("prepared", "result", "stage"):
        st.session_state.pop(key, None)
    st.session_state.stage = 1


def risk_banner(level: str, title: str) -> None:
    message = f"{title}：{level}风险"
    if level == "高":
        st.error(message)
    elif level == "中":
        st.warning(message)
    else:
        st.success(message)


def render_comments(comments) -> None:
    if not comments:
        st.info("本轮没有生成可显示的评论。")
        return
    for comment in comments:
        prefix = "↳ 回复" if comment.parent_id else "评论"
        with st.container(border=True):
            st.markdown(
                f'<div class="comment-meta">{prefix} · {comment.persona_label} · '
                f'第{comment.round_no}轮 · 赞 {comment.likes}</div>'
                f'<div class="comment-text">{comment.text}</div>',
                unsafe_allow_html=True,
            )


def render_risk(report) -> None:
    risk_banner(report.overall_level, "综合判断")
    st.write(report.summary)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("高风险原文")
        for span in report.risky_spans:
            st.markdown(f"**“{span.text}”**  ")
            st.caption(span.reason)
    with col2:
        st.subheader("修改方向")
        for index, direction in enumerate(report.modification_directions, 1):
            st.write(f"{index}. {direction}")
    st.subheader("可能的误解链")
    for chain in report.misunderstanding_chains:
        st.write(f"- {chain}")


def render_comparison(result: ProjectResult) -> None:
    before, after = result.risk_before, result.risk_after
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("修改前", f"{before.overall_level}风险")
    c2.metric("修改后", f"{after.overall_level}风险")
    c3.markdown(f"**结论**  \n{result.comparison.conclusion}")
    metrics = result.comparison
    cols = st.columns(4)
    cols[0].metric("误解评论变化", f"{metrics.misunderstanding_change:+.1%}")
    cols[1].metric("负面评论变化", f"{metrics.negative_change:+.1%}")
    cols[2].metric("对立回复变化", f"{metrics.conflict_change:+d}")
    cols[3].metric("跑题评论变化", f"{metrics.off_topic_change:+.1%}")
    st.caption("以上是同一批模拟受众下的压力测试差值，不代表真实平台概率。")


orchestrator = get_orchestrator()
st.session_state.setdefault("stage", 1)

with st.sidebar:
    st.markdown("### CommentLab")
    st.caption("发布前沟通风险模拟")
    mode_label = "DEMO_MODE（无密钥可运行）" if orchestrator.demo_mode else f"实时模型：{settings.model}"
    st.info(mode_label)
    if orchestrator.gateway.last_error:
        st.warning("最近一次模型调用失败，系统已使用本地降级结果完成流程。")
    if st.button("新建测试", use_container_width=True):
        reset_flow()
        st.rerun()
    st.divider()
    st.markdown("#### 历史结果")
    history = orchestrator.database.list_projects()
    if history:
        selected_id = st.selectbox(
            "选择历史项目",
            options=[item.project_id for item in history],
            format_func=lambda project_id: next(
                f"{item.created_at[:16]} · {item.post_text[:18]}"
                for item in history
                if item.project_id == project_id
            ),
            label_visibility="collapsed",
        )
        if st.button("打开历史结果", use_container_width=True):
            loaded = orchestrator.database.load_project(selected_id)
            if loaded:
                st.session_state.result = loaded
                st.session_state.stage = 4
                st.rerun()
    else:
        st.caption("还没有保存的测试。")

st.title("CommentLab")
st.markdown("#### 面向个人创作者的发布前沟通鲁棒性测试 Agent")
st.markdown(
    '<div class="disclaimer">本系统用于发布前沟通风险压力测试，模拟结果不代表真实舆情预测。</div>',
    unsafe_allow_html=True,
)
st.write("")

steps = ["1 输入", "2 受众确认", "3 原文结果", "4 改写对比"]
st.progress(st.session_state.stage / 4, text=steps[st.session_state.stage - 1])

if st.session_state.stage == 1:
    cases = load_demo_cases()
    case_names = ["自定义"] + [case["name"] for case in cases]
    st.session_state.setdefault("input_post", "")
    st.session_state.setdefault("input_identity", "个人内容创作者")
    st.session_state.setdefault("input_domain", "日常分享")
    st.session_state.setdefault("input_follower_scale", "中小体量")
    st.session_state.setdefault("input_style", "理性、直接")
    st.session_state.setdefault("input_audience_relationship", "普通关注关系")
    st.selectbox(
        "课堂演示案例",
        case_names,
        key="case_selector",
        on_change=apply_demo_case,
    )
    with st.form("input_form"):
        post_text = st.text_area(
            "准备发布的帖子",
            key="input_post",
            max_chars=500,
            height=150,
            help="仅支持500字以内中文短文本；第一版不读取链接、图片、视频或历史内容。",
        )
        st.caption(f"当前 {len(post_text)} / 500 字")
        st.subheader("发布者画像")
        col1, col2, col3 = st.columns(3)
        identity = col1.text_input("身份", key="input_identity")
        domain = col2.text_input("内容领域", key="input_domain")
        follower_scale = col3.selectbox(
            "粉丝规模",
            ["小体量", "中小体量", "中等体量", "较大体量"],
            key="input_follower_scale",
        )
        style = st.text_input("表达风格", key="input_style")
        audience_relationship = st.text_input("与受众关系", key="input_audience_relationship")
        submitted = st.form_submit_button("分析内容并生成受众", type="primary")
    if submitted:
        try:
            profile = PublisherProfile(
                identity=identity,
                domain=domain,
                follower_scale=follower_scale,
                style=style,
                audience_relationship=audience_relationship,
            )
            with st.spinner("内容分析 Agent 与受众规划 Agent 正在工作..."):
                st.session_state.prepared = orchestrator.prepare(post_text, profile)
            st.session_state.stage = 2
            st.rerun()
        except Exception as exc:
            st.error(f"无法开始分析：{exc}")

elif st.session_state.stage == 2:
    prepared = st.session_state.get("prepared")
    if prepared is None:
        reset_flow()
        st.rerun()
    st.subheader("内容初步分析")
    st.write(f"**核心表达：** {prepared.analysis.main_message}")
    if prepared.analysis.ambiguous_phrases:
        st.write("**可能模糊的表达：**")
        for issue in prepared.analysis.ambiguous_phrases:
            st.write(f"- “{issue.text}”：{issue.reason}")
    if prepared.analysis.missing_information:
        st.write("**可能缺失的信息：** " + "、".join(prepared.analysis.missing_information))
    st.divider()
    st.subheader("确认模拟受众")
    st.caption("先调大类比例；需要时再展开修改具体Persona权重。系统会自动归一化。")
    with st.form("audience_form"):
        group_values = {}
        group_columns = st.columns(3)
        for index, (group, ratio) in enumerate(prepared.audience.group_ratios.items()):
            group_values[group] = group_columns[index % 3].slider(
                group, 0, 100, int(round(ratio * 100)), 5
            )
        persona_weights = {}
        with st.expander("高级：具体 Persona 权重"):
            for persona in prepared.audience.personas:
                if not persona.active:
                    st.caption(f"{persona.label}：由Python规则模拟，不参与发言权重")
                    continue
                persona_weights[persona.persona_id] = st.slider(
                    f"{persona.label} · {persona.description}",
                    0.0,
                    3.0,
                    float(min(3.0, persona.weight * 10)),
                    0.1,
                )
        start = st.form_submit_button("开始三轮模拟", type="primary")
    if start:
        personas = [persona.model_copy(deep=True) for persona in prepared.audience.personas]
        for persona in personas:
            if persona.persona_id in persona_weights:
                persona.weight = persona_weights[persona.persona_id]
        audience = AudiencePlan(
            personas=personas,
            group_ratios={key: value / 100 for key, value in group_values.items()},
            rationale=prepared.audience.rationale,
        ).normalized()
        try:
            with st.spinner("受众 Agent 正在进行原文与改写后三轮反事实模拟..."):
                st.session_state.result = orchestrator.complete(prepared, audience=audience, seed=42)
            st.session_state.stage = 3
            st.rerun()
        except Exception as exc:
            st.error(f"模拟未完成：{exc}")

elif st.session_state.stage == 3:
    result: ProjectResult | None = st.session_state.get("result")
    if result is None:
        reset_flow()
        st.rerun()
    st.header("原文压力测试")
    st.markdown(f"> {result.post_text}")
    render_risk(result.risk_before)
    st.divider()
    st.subheader("模拟评论区")
    render_comments(result.simulation_before.comments)
    if st.button("查看改写与反事实对比", type="primary"):
        st.session_state.stage = 4
        st.rerun()

else:
    result: ProjectResult | None = st.session_state.get("result")
    if result is None:
        reset_flow()
        st.rerun()
    st.header("改写与反事实对比")
    render_comparison(result)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原文")
        st.markdown(f"> {result.post_text}")
        risk_banner(result.risk_before.overall_level, "修改前")
    with col2:
        st.subheader("保持人设的改写")
        st.markdown(f"> {result.rewrite.rewritten_post}")
        risk_banner(result.risk_after.overall_level, "修改后")
        st.caption(result.rewrite.explanation)
    st.subheader("改写后模拟评论区")
    render_comments(result.simulation_after.comments)
    st.divider()
    st.subheader("仍需注意")
    for item in result.comparison.remaining_questions:
        st.write(f"- {item}")
    st.caption(f"同一受众配置验证：{'通过' if result.comparison.persona_consistency else '未通过'}")
