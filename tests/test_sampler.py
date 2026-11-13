"""The sampler: flow times and the spec strings the scripts take. CPU only."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.model.flow import flow_times, parse_spec, sampler_kwargs

CFG = {"cfg_audio": 2.0, "sample_steps": 4, "frames": 16, "prefix": 4}


def test_flow_times():
    assert flow_times(4) == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


def test_specs():
    assert parse_spec("g=2,steps=6") == {"g": 2.0, "steps": 6.0} and parse_spec("") == {}
    for bad in ("g=2,mouht=1.25", "mouth=1.25", "tau=0.5", "avg=4", "ramp=1.5"):
        with pytest.raises(ValueError):
            parse_spec(bad)
    assert sampler_kwargs(parse_spec("g=3,steps=6"), CFG) == {"cfg_audio": 3.0, "steps": 6}
    assert sampler_kwargs({}, CFG) == {"cfg_audio": 2.0, "steps": 4}
