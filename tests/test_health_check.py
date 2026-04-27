from __future__ import annotations

from scripts.health_check import _parse_step_lines


def test_health_check_parses_ppo_critic_step_lines():
    lines = [
        "[ppo_critic step 12/1000] loss=1.25 pg=1.0 reward=0.125 pass@G=0.500\n",
    ]

    parsed = _parse_step_lines(lines)

    assert parsed == [(12, 1.25, 0.125, 0.5)]
