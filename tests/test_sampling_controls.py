import pytest

torch = pytest.importorskip("torch")

from drift.orchestrator import Orchestrator  # noqa: E402


def picker():
    return Orchestrator.__new__(Orchestrator)


def test_default_picker_is_greedy_argmax():
    orch = picker()
    logits = torch.tensor([0.1, 0.2, 4.0, 0.3])

    assert orch._pick_token(logits, [], None, 0) == 2
    assert orch._pick_token(logits, [], {"temperature": 0.0}, 0) == 2


def test_penalty_can_change_temperature_zero_greedy_choice():
    orch = picker()
    logits = torch.tensor([0.1, 0.2, 4.0, 3.8])

    got = orch._pick_token(
        logits,
        [2],
        {"temperature": 0.0, "presence_penalty": 1.0, "frequency_penalty": 0.0},
        0,
    )

    assert got == 3


def test_seeded_sampling_is_repeatable():
    orch = picker()
    logits = torch.tensor([1.0, 1.1, 1.2, 1.3])
    opts = {"temperature": 0.8, "top_p": 0.95, "seed": 7}

    first = [orch._pick_token(logits, [], opts, step) for step in range(5)]
    second = [orch._pick_token(logits, [], opts, step) for step in range(5)]

    assert first == second
