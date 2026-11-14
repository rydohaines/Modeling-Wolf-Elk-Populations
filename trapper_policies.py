"""Trapper-policy extensions built on top of ``ecosystem_ode``.

Imports the shared base model (``ODESystem``, ``WOLF_ELK_DEFAULTS``,
``wolf_elk_rhs``, …) and adds wolf-only trapping scenarios:

- seasonal open-season pulses (``make_seasonal_trapper_model``)
- lagged census policies, with optional observation noise
  (``solve_trapper_policy``)

Defaults set ``F=0`` so these scenarios isolate trapping. Biological
parameters otherwise inherit from ``WOLF_ELK_DEFAULTS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecosystem_ode import (
    ODESystem,
    WOLF_ELK_DEFAULTS,
    wolf_elk_rhs,
)


TrapperPolicyName = Literal["adaptive", "noisy_adaptive"]


def periodic_pulse(
    t: float,
    period: float,
    width: float,
    height: float,
    base: float = 0.0,
) -> float:
    """Rectangular periodic pulse: base most of the time, base+height for width each period."""
    if period <= 0:
        raise ValueError("period must be strictly positive")
    if width < 0 or width > period:
        raise ValueError("width must satisfy 0 <= width <= period")
    phase = float(t) % period
    if phase < width:
        return base + height
    return base


def trapper_pressure(t: float, p: Mapping[str, float]) -> float:
    """Exogenous trapper effort Tr(t); seasonal pulse if pulse params are present."""
    pulse_keys = ("Tr_base", "Tr_pulse", "T", "pulse_width")
    if all(key in p for key in pulse_keys):
        return periodic_pulse(
            t, period=p["T"], width=p["pulse_width"],
            height=p["Tr_pulse"], base=p["Tr_base"],
        )
    return float(p.get("Tr", 0.0))


def adaptive_trapper_effort(W_prev: float, p: Mapping[str, float]) -> float:
    """Lagged abundance policy: raise effort when last wolf count exceeds a target.

    Tr = clip(Tr_off + kappa * (W_prev - W_target), 0, Tr_max)
    """
    W_prev = max(0.0, float(W_prev))
    effort = p["Tr_off"] + p["kappa"] * (W_prev - p["W_target"])
    return float(np.clip(effort, 0.0, p["Tr_max"]))


def observe_wolf_census(
    W_true: float,
    sigma_obs: float,
    rng: np.random.Generator,
) -> float:
    """Lognormal census noise (median-unbiased): W_obs = W * exp(sigma Z - sigma^2/2)."""
    W_true = max(0.0, float(W_true))
    if sigma_obs <= 0.0 or W_true == 0.0:
        return W_true
    return float(W_true * np.exp(sigma_obs * rng.normal() - 0.5 * sigma_obs**2))


def noisy_adaptive_trapper_effort(
    W_prev: float,
    p: Mapping[str, float],
    rng: np.random.Generator,
) -> float:
    """Identical target rule as ``adaptive_trapper_effort``, on a noisy census."""
    sigma = float(p.get("sigma_obs", p.get("sigma_Tr", 0.0)))
    W_obs = observe_wolf_census(W_prev, sigma, rng)
    return adaptive_trapper_effort(W_obs, p)


def wolf_elk_derivatives(
    t: float,
    y: NDArray[np.float64],
    p: Mapping[str, float],
    Tr: float,
) -> NDArray[np.float64]:
    """Base wolf–elk field from ``ecosystem_ode``, plus wolf-only trapper mortality."""
    dE, dW = wolf_elk_rhs(t, y, p)
    W = float(y[1])
    trapper_wolf_loss = float(p.get("c_TW", 0.0)) * float(Tr) * W
    return np.array([dE, dW - trapper_wolf_loss], dtype=float)


def wolf_elk_trapper_rhs(
    t: float,
    y: NDArray[np.float64],
    p: Mapping[str, float],
) -> NDArray[np.float64]:
    """Extended RHS: inherit base farmers/hunters dynamics, add Tr(t) on wolves."""
    return wolf_elk_derivatives(t, y, p, trapper_pressure(t, p))


# Inherit shared biology; trapper scenarios turn farmers off by default.
SEASONAL_TRAPPER_DEFAULTS: dict[str, float] = {
    **WOLF_ELK_DEFAULTS,
    "F": 0.0,
    "Tr_base": 0.0,
    "Tr_pulse": 90.0,
    "T": 15.0,
    "pulse_width": 4.0,
    "c_TW": 0.02,
    "Tr": 0.0,
}

ADAPTIVE_TRAPPER_DEFAULTS: dict[str, float] = {
    **WOLF_ELK_DEFAULTS,
    "F": 0.0,
    "c_TW": 0.02,
    "W_target": 25.0,
    "Tr_off": 1.0,
    "kappa": 0.45,
    "Tr_max": 55.0,
    "decision_period": 5.0,
    "W_extinct": 0.5,
    "sigma_obs": 1.0,
    "sigma_Tr": 1.0,
}


def _validate_nonnegative_capacity(parameters: Mapping[str, float]) -> None:
    if parameters["K"] <= 0:
        raise ValueError("K must be strictly positive")
    if parameters["K"] - parameters["alpha_FE"] * parameters["F"] <= 0:
        raise ValueError("Farmer pressure must leave a positive elk carrying capacity")
    if any(v < 0 for v in parameters.values()):
        raise ValueError("Parameters must be nonnegative")


def make_seasonal_trapper_model(**parameter_overrides: float) -> ODESystem:
    """Seasonal trapper pulses on wolves only; inherits ``WOLF_ELK_DEFAULTS`` with F=0."""
    invalid = set(parameter_overrides) - set(SEASONAL_TRAPPER_DEFAULTS)
    if invalid:
        raise KeyError(f"Unknown seasonal-trapper parameter(s): {sorted(invalid)}")
    parameters = {**SEASONAL_TRAPPER_DEFAULTS, **parameter_overrides}
    period = parameters["T"]
    width = parameters["pulse_width"]
    if period <= 0:
        raise ValueError("T must be strictly positive")
    if width < 0 or width > period:
        raise ValueError("pulse_width must satisfy 0 <= pulse_width <= T")
    _validate_nonnegative_capacity(parameters)
    return ODESystem(
        rhs=wolf_elk_trapper_rhs,
        state_names=("Elk", "Wolves"),
        parameters=parameters,
        title="Wolf–elk ecosystem (seasonal trappers)",
    )


@dataclass
class TrappingRun:
    """Fixed-step simulation with an explicit trapper-effort time series."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    tr: NDArray[np.float64]
    parameters: dict[str, float]
    policy: str
    title: str = "Wolf–elk ecosystem (trapper policy)"

    @property
    def success(self) -> bool:
        return bool(np.all(np.isfinite(self.y)))


def _rk4_step(
    t: float,
    y: NDArray[np.float64],
    dt: float,
    p: Mapping[str, float],
    Tr: float,
) -> NDArray[np.float64]:
    k1 = wolf_elk_derivatives(t, y, p, Tr)
    k2 = wolf_elk_derivatives(t + 0.5 * dt, y + 0.5 * dt * k1, p, Tr)
    k3 = wolf_elk_derivatives(t + 0.5 * dt, y + 0.5 * dt * k2, p, Tr)
    k4 = wolf_elk_derivatives(t + dt, y + dt * k3, p, Tr)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_trapper_policy(
    y0: ArrayLike,
    t_span: tuple[float, float] = (0.0, 400.0),
    *,
    policy: TrapperPolicyName = "adaptive",
    dt: float = 0.05,
    seed: int | None = None,
    parameters: Mapping[str, float] | None = None,
    **parameter_overrides: float,
) -> TrappingRun:
    """Lagged-census trapping on top of the base ODE; F=0 by default.

    - ``adaptive``: stable positive stock under default knobs
    - ``noisy_adaptive``: same rule on a noisy census; can crash wolves
    """
    if policy not in ("adaptive", "noisy_adaptive"):
        raise ValueError("policy must be 'adaptive' or 'noisy_adaptive'")
    if dt <= 0:
        raise ValueError("dt must be strictly positive")
    if t_span[1] <= t_span[0]:
        raise ValueError("t_span must have increasing start and end times")

    params = {**ADAPTIVE_TRAPPER_DEFAULTS, **dict(parameters or {}), **parameter_overrides}
    if "sigma_obs" in parameter_overrides and "sigma_Tr" not in parameter_overrides:
        params["sigma_Tr"] = params["sigma_obs"]
    if "sigma_Tr" in parameter_overrides and "sigma_obs" not in parameter_overrides:
        params["sigma_obs"] = params["sigma_Tr"]
    for key in ("Tr_base", "Tr_pulse", "T", "pulse_width"):
        params.pop(key, None)
    invalid = set(params) - set(ADAPTIVE_TRAPPER_DEFAULTS)
    if invalid:
        raise KeyError(f"Unknown adaptive-trapper parameter(s): {sorted(invalid)}")
    _validate_nonnegative_capacity(params)
    if params["decision_period"] <= 0:
        raise ValueError("decision_period must be strictly positive")
    if policy == "noisy_adaptive" and params["sigma_obs"] < 0:
        raise ValueError("sigma_obs must be nonnegative")

    initial = np.asarray(y0, dtype=float)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("y0 must be a finite length-2 vector [elk, wolves]")

    times = np.arange(t_span[0], t_span[1] + 0.5 * dt, dt)
    if times[-1] > t_span[1]:
        times = times[times <= t_span[1]]
    if len(times) < 2 or abs(times[-1] - t_span[1]) > 1e-12:
        times = np.append(times[times < t_span[1]], t_span[1])

    states = np.zeros((2, len(times)))
    efforts = np.zeros(len(times))
    states[:, 0] = initial
    rng = np.random.default_rng(seed)
    period_steps = max(1, int(round(params["decision_period"] / dt)))
    W_extinct = float(params["W_extinct"])
    prev_census = float(initial[1])
    i = 0

    while i < len(times) - 1:
        if states[1, i] <= W_extinct:
            states[1, i:] = 0.0
            efforts[i:] = 0.0
            break

        if policy == "adaptive":
            Tr = adaptive_trapper_effort(prev_census, params)
        else:
            Tr = noisy_adaptive_trapper_effort(prev_census, params, rng)

        for _ in range(period_steps):
            if i >= len(times) - 1:
                break
            efforts[i] = Tr
            step = times[i + 1] - times[i]
            nxt = np.maximum(_rk4_step(times[i], states[:, i], step, params, Tr), 0.0)
            if nxt[1] <= W_extinct:
                nxt[1] = 0.0
            states[:, i + 1] = nxt
            i += 1
            if states[1, i] <= W_extinct:
                states[1, i:] = 0.0
                efforts[i:] = 0.0
                i = len(times) - 1
                break

        prev_census = float(states[1, min(i, len(times) - 1)])

    if efforts[-1] == 0.0 and len(times) > 1 and states[1, -1] > W_extinct:
        efforts[-1] = efforts[-2]

    title = {
        "adaptive": "Wolf–elk ecosystem (adaptive trappers, no farmers)",
        "noisy_adaptive": "Wolf–elk ecosystem (noisy census trappers, no farmers)",
    }[policy]
    return TrappingRun(
        t=times, y=states, tr=efforts, parameters=params, policy=policy, title=title,
    )


__all__ = [
    "ADAPTIVE_TRAPPER_DEFAULTS",
    "SEASONAL_TRAPPER_DEFAULTS",
    "TrappingRun",
    "adaptive_trapper_effort",
    "make_seasonal_trapper_model",
    "noisy_adaptive_trapper_effort",
    "observe_wolf_census",
    "periodic_pulse",
    "solve_trapper_policy",
    "trapper_pressure",
    "wolf_elk_derivatives",
    "wolf_elk_trapper_rhs",
]
