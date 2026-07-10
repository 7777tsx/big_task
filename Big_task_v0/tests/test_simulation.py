from __future__ import annotations

from models.schemas import Comment, CommentAction, SimulationConfig
from simulation.engine import SimulationEngine
from simulation.heat import calculate_hot_score, simulate_lurker_likes


def sample_comment(comment_id: str = "c1") -> Comment:
    return Comment(
        comment_id=comment_id,
        persona_id="p1",
        persona_label="测试Persona",
        round_no=1,
        text="这是一条满足长度要求的测试评论内容。",
        stance="question",
        emotion="疑惑",
        emotion_intensity=0.5,
        controversy=0.4,
    )


def test_hot_score_formula() -> None:
    score = calculate_hot_score(10, 2, 0.5, 0.4, 2.0)
    assert score == 7.3


def test_lurker_likes_are_deterministic(orchestrator, profile) -> None:
    audience = orchestrator.audience_chain._default_plan()
    first = [sample_comment()]
    second = [sample_comment()]
    simulate_lurker_likes(first, audience.personas, 50, 42, 1)
    simulate_lurker_likes(second, audience.personas, 50, 42, 1)
    assert first[0].likes == second[0].likes
    assert first[0].hot_score == second[0].hot_score


def test_invalid_reply_target_becomes_top_level(orchestrator) -> None:
    audience = orchestrator.audience_chain._default_plan()
    persona = next(persona for persona in audience.personas if persona.active)
    action = CommentAction(
        persona_id=persona.persona_id,
        action="reply",
        target_comment_id="missing",
        text="目标不存在时这条回复应该转换为普通评论。",
        stance="question",
    )
    comments = []
    created = SimulationEngine._apply_actions(
        [action], comments, [persona], 2, "before"
    )
    assert created == 1
    assert comments[0].parent_id is None


def test_three_round_simulation_is_repeatable(orchestrator, profile) -> None:
    prepared = orchestrator.prepare("最近会适当减少更新，大家不用多想。", profile)
    config = SimulationConfig(post_text=prepared.post_text, seed=7)
    first = orchestrator.simulation_engine.run(config, profile, prepared.audience)
    second = orchestrator.simulation_engine.run(config, profile, prepared.audience)
    assert len(first.comments) == 14
    assert 12 <= len(first.comments) <= 20
    assert first.model_dump() == second.model_dump()
    assert {event.round_no for event in first.trace} == {1, 2, 3}
    assert len({comment.persona_id for comment in first.comments}) == 10
