import math

import pytest

from rleval.actions import NEUTRAL, parse_action


@pytest.mark.parametrize(
    "name,value",
    [
        ("throttle", math.nan),
        ("steer", math.inf),
        ("yaw", 1.1),
        ("roll", True),
        ("jump", 1),
        ("boost", "false"),
    ],
)
def test_rejects_unsafe_controller_values(name, value):
    with pytest.raises(ValueError):
        parse_action({"controls": dict(NEUTRAL, **{name: value}), "hold_ms": 100, "memory": ""})


@pytest.mark.parametrize("hold", [True, -1, 0, 1001, 100.0])
def test_rejects_invalid_action_lifetimes(hold):
    with pytest.raises(ValueError):
        parse_action({"controls": dict(NEUTRAL), "hold_ms": hold, "memory": ""})


def test_no_arbitrary_action_fields():
    with pytest.raises(ValueError):
        parse_action({"controls": NEUTRAL, "hold_ms": 100, "memory": "", "command": "anything"})
