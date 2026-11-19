"""Time-dependent introductions built on Norman's Beddington–DeAngelis model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_model_snapshot import WOLF_ELK_DEFAULTS, wolf_elk_rhs

MODEL_SOURCE = "Norman Beddington–DeAngelis model"

SHAPES = ("exponential", "linear", "smoothstep")
SCENARIOS = ("control", "farmer", "hunter", "both")


def ramp(t, shape="linear", start=10.0, duration=35.0):
    """Normalized 0-to-1 introduction with bounded endpoints."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    u = np.clip((np.asarray(t, dtype=float) - start) / duration, 0.0, 1.0)
    if shape == "linear":
        value = u
    elif shape == "exponential":
        value = np.expm1(4 * u) / np.expm1(4)
    elif shape == "smoothstep":
        value = 3 * u**2 - 2 * u**3
    else:
        raise ValueError(f"Unknown ramp shape: {shape}")
    return float(value) if np.ndim(value) == 0 else value


def pressures(t, scenario, shape, max_farmers, max_hunters, start, duration):
    """Return actual farmer and hunter pressures, not fractions."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    level = ramp(t, shape=shape, start=start, duration=duration)
    farmers = max_farmers * level if scenario in ("farmer", "both") else 0.0
    hunters = max_hunters * level if scenario in ("hunter", "both") else 0.0
    return float(farmers), float(hunters)


def make_rhs(scenario="both", shape="linear", *, max_farmers=40.0,
             max_hunters=25.0, start=10.0, duration=35.0,
             parameters=None):
    """Apply time-varying farmer and hunter pressure to the BD model."""
    p = dict(WOLF_ELK_DEFAULTS)
    if parameters:
        unknown = set(parameters) - set(WOLF_ELK_DEFAULTS)
        if unknown:
            raise KeyError(f"Unknown original parameters: {sorted(unknown)}")
        p.update(parameters)
    if max_farmers < 0 or max_hunters < 0:
        raise ValueError("Maximum pressures must be nonnegative")
    if p["K"] - p["alpha_FE"] * max_farmers <= 0:
        raise ValueError("K - alpha_FE * max_farmers must stay positive")

    def rhs(t, y):
        farmers, hunters = pressures(
            t, scenario, shape, max_farmers, max_hunters, start, duration
        )
        local = {**p, "F": farmers, "H": hunters}
        return wolf_elk_rhs(t, np.asarray(y), local)

    return rhs


def simulate(initial=(600, 50), scenario="both", shape="linear",
             t_span=(0.0, 120.0), n_points=700, **kwargs):
    """Integrate one nonnegative (elk, wolves) initial state."""
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (2,) or np.any(initial < 0):
        raise ValueError("Initial (elk, wolf) populations must be nonnegative")
    if n_points < 2 or t_span[1] <= t_span[0]:
        raise ValueError("Use an increasing time span and at least two points")
    rhs = make_rhs(scenario=scenario, shape=shape, **kwargs)
    result = solve_ivp(
        rhs,
        t_span,
        initial,
        t_eval=np.linspace(*t_span, n_points),
        rtol=1e-8,
        atol=1e-10,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result
