from __future__ import annotations

import scripts.run_competition_scan as competition_runner
from competition_core.sequence_mining import SequenceCandidate
from scripts.run_competition_scan import (
    _candidate_interactions,
    _console_safe,
    _probe_event_relay,
)


def test_candidate_interactions_align_each_forward_input_with_next_token() -> None:
    candidate = SequenceCandidate(
        token_ids=(11, 12, 13),
        text=" first second third",
        continuation_probabilities=(0.8, 0.9),
        suffix_floor=0.8,
        mean_log_probability=-0.2,
        used_beam=True,
        seed_token_id=11,
        token_texts=(" first", " second", " third"),
        selection_modes=("beam_search", "greedy"),
    )

    interactions = _candidate_interactions(
        candidate,
        response_prefix="### Response:\n",
    )

    assert interactions == [
        {
            "step": 1,
            "input_text": "### Response:\n first",
            "input_token_ids": [11],
            "output_token_id": 12,
            "output_token_text": " second",
            "output_probability": 0.8,
            "selection_mode": "beam_search",
        },
        {
            "step": 2,
            "input_text": "### Response:\n first second",
            "input_token_ids": [11, 12],
            "output_token_id": 13,
            "output_token_text": " third",
            "output_probability": 0.9,
            "selection_mode": "greedy",
        },
    ]


def test_probe_event_relay_batches_steps_without_dropping_inputs(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        competition_runner,
        "emit",
        lambda event_type, **payload: events.append({"type": event_type, **payload}),
    )
    on_line, flush = _probe_event_relay(
        candidate_count=4,
        batch_size=8,
        soft_token_count=12,
    )

    on_line('[latent-probe-inputs] {"inputs":[{"index":0,"text":"prompt"}]}')
    for step in range(1, 10):
        on_line(
            '[latent-probe-step] {"rank":1,"step":{"step":'
            + str(step)
            + "}}"
        )
    on_line(
        '[latent-replay] {"rank":1,"replay":'
        '{"soft_trigger_exact_prefix_match_rate":0.75,"examples":[]},'
        '"replay_refinement":{"used":true,"decision_use":false}}'
    )
    on_line(
        "[latent-probe] rank=1 max_gap=0.400000 "
        "max_log_gap=1.500000 criterion_met=True"
    )
    flush()

    assert events[0]["type"] == "competition_probe_inputs"
    step_events = [event for event in events if event["type"] == "competition_probe_steps"]
    assert [len(event["steps"]) for event in step_events] == [8, 1]
    assert [item["step"] for event in step_events for item in event["steps"]] == list(
        range(1, 10)
    )
    replay_event = next(
        event for event in events if event["type"] == "competition_soft_replay"
    )
    assert replay_event["replay"]["soft_trigger_exact_prefix_match_rate"] == 0.75
    assert replay_event["replay_refinement"]["used"] is True
    progress_event = next(
        event for event in events if event["type"] == "competition_probe_progress"
    )
    assert progress_event["max_log_likelihood_gap"] == 1.5


def test_console_safe_replaces_characters_that_gbk_cannot_encode() -> None:
    assert _console_safe("Loading weights: \ufffd", encoding="gbk") == "Loading weights: ?"
