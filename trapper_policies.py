"""Trapper-policy extensions built on top of ``ecosystem_ode``.

Imports the shared base model (``ODESystem``, ``WOLF_ELK_DEFAULTS``,
``wolf_elk_rhs``, …) and adds wolf-only trapping scenarios:

- seasonal open-season pulses (``make_seasonal_trapper_model``)
- lagged census hunting policies, with optional untracked extra removals
  (``solve_trapper_policy``)

Noise is applied to wolf removals (untracked hunting), not to the census.
Defaults set ``F=0`` so these scenarios isolate hunting policy effects.
Biological parameters otherwise inherit from ``WOLF_ELK_DEFAULTS``.
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


TrapperPolicyName = Literal["adaptive", "untracked", "noisy_adaptive"]


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
    """Tracked hunting policy from the true lagged wolf census.

    Tr_policy = clip(Tr_off + kappa * (W_prev - W_target), 0, Tr_max)
    """
    W_prev = max(0.0, float(W_prev))
    effort = p["Tr_off"] + p["kappa"] * (W_prev - p["W_target"])
    return float(np.clip(effort, 0.0, p["Tr_max"]))


def untracked_hunting_effort(
    p: Mapping[str, float],
    rng: np.random.Generator,
) -> float:
    """Nonnegative untracked/extra wolf-removal effort for one decision window.

    Drawn as a median-unbiased lognormal around ``Tr_untracked`` with scale
    ``sigma_untracked``. This is hunting that is not part of the tracked policy.
    """
    mean = float(p.get("Tr_untracked", 0.0))
    sigma = float(p.get("sigma_untracked", p.get("sigma_Tr", 0.0)))
    if mean <= 0.0:
        return 0.0
    if sigma <= 0.0:
        return mean
    return float(mean * np.exp(sigma * rng.normal() - 0.5 * sigma**2))


def total_hunting_effort(
    W_prev: float,
    p: Mapping[str, float],
    *,
    include_untracked: bool,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (Tr_total, Tr_policy, Tr_untracked) for one decision window."""
    policy = adaptive_trapper_effort(W_prev, p)
    if not include_untracked:
        return policy, policy, 0.0
    if rng is None:
        raise ValueError("rng is required when include_untracked is True")
    extra = untracked_hunting_effort(p, rng)
    total = float(np.clip(policy + extra, 0.0, p.get("Tr_total_max", policy + extra)))
    return total, policy, extra


def wolf_elk_derivatives(
    t: float,
    y: NDArray[np.float64],
    p: Mapping[str, float],
    Tr: float,
) -> NDArray[np.float64]:
    """Base wolf–elk field from ``ecosystem_ode``, plus wolf-only hunting mortality."""
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

# Tracked adaptive hunting is stable; adding untracked removals can crash wolves.
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
    "Tr_untracked": 10.0,     # typical untracked extra effort each window
    "sigma_untracked": 0.40,  # noise scale on untracked effort (not on census)
    "Tr_total_max": 80.0,     # cap on policy + untracked combined
    "sigma_Tr": 0.40,         # alias of sigma_untracked
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
    """Fixed-step simulation with tracked and optional untracked hunting effort."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    tr: NDArray[np.float64]
    tr_policy: NDArray[np.float64]
    tr_untracked: NDArray[np.float64]
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
    """Integrate hunting policy effects on wolves (F=0 by default).

    Every ``decision_period``, tracked effort is set from the true lagged wolf
    census. Policies:

    - ``adaptive``: tracked policy only
    - ``untracked`` / ``noisy_adaptive``: same tracked policy plus random
      untracked wolf removals (noise on harvest, not on the census)
    """
    if policy == "noisy_adaptive":
        policy = "untracked"
    if policy not in ("adaptive", "untracked"):
        raise ValueError("policy must be 'adaptive' or 'untracked'")
    if dt <= 0:
        raise ValueError("dt must be strictly positive")
    if t_span[1] <= t_span[0]:
        raise ValueError("t_span must have increasing start and end times")

    params = {**ADAPTIVE_TRAPPER_DEFAULTS, **dict(parameters or {}), **parameter_overrides}
    if "sigma_untracked" in parameter_overrides and "sigma_Tr" not in parameter_overrides:
        params["sigma_Tr"] = params["sigma_untracked"]
    if "sigma_Tr" in parameter_overrides and "sigma_untracked" not in parameter_overrides:
        params["sigma_untracked"] = params["sigma_Tr"]
    for key in ("Tr_base", "Tr_pulse", "T", "pulse_width"):
        params.pop(key, None)
    # Drop obsolete census-noise key if present in overrides/old callers.
    params.pop("sigma_obs", None)
    invalid = set(params) - set(ADAPTIVE_TRAPPER_DEFAULTS)
    if invalid:
        raise KeyError(f"Unknown adaptive-trapper parameter(s): {sorted(invalid)}")
    _validate_nonnegative_capacity(params)
    if params["decision_period"] <= 0:
        raise ValueError("decision_period must be strictly positive")
    if policy == "untracked" and params["sigma_untracked"] < 0:
        raise ValueError("sigma_untracked must be nonnegative")

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
    efforts_policy = np.zeros(len(times))
    efforts_untracked = np.zeros(len(times))
    states[:, 0] = initial
    rng = np.random.default_rng(seed)
    period_steps = max(1, int(round(params["decision_period"] / dt)))
    W_extinct = float(params["W_extinct"])
    prev_census = float(initial[1])
    include_untracked = policy == "untracked"
    i = 0

    extinct = False
    while i < len(times) - 1:
        if states[1, i] <= W_extinct:
            states[1, i] = 0.0
            extinct = True

        if extinct:
            # Wolves stay extinct; keep integrating elk with no hunting pressure.
            Tr, Tr_pol, Tr_extra = 0.0, 0.0, 0.0
            prev_census = 0.0
        else:
            Tr, Tr_pol, Tr_extra = total_hunting_effort(
                prev_census, params, include_untracked=include_untracked, rng=rng,
            )

        for _ in range(period_steps):
            if i >= len(times) - 1:
                break
            efforts[i] = Tr
            efforts_policy[i] = Tr_pol
            efforts_untracked[i] = Tr_extra
            step = times[i + 1] - times[i]
            state = states[:, i].copy()
            if extinct:
                state[1] = 0.0
            nxt = np.maximum(_rk4_step(times[i], state, step, params, Tr), 0.0)
            if nxt[1] <= W_extinct:
                nxt[1] = 0.0
                extinct = True
                Tr, Tr_pol, Tr_extra = 0.0, 0.0, 0.0
            states[:, i + 1] = nxt
            i += 1

        prev_census = float(states[1, min(i, len(times) - 1)])

    if efforts[-1] == 0.0 and len(times) > 1 and states[1, -1] > W_extinct:
        efforts[-1] = efforts[-2]
        efforts_policy[-1] = efforts_policy[-2]
        efforts_untracked[-1] = efforts_untracked[-2]

    title = {
        "adaptive": "Tracked adaptive wolf hunting (no untracked harvest)",
        "untracked": "Tracked adaptive policy + untracked wolf hunting",
    }[policy]
    return TrappingRun(
        t=times,
        y=states,
        tr=efforts,
        tr_policy=efforts_policy,
        tr_untracked=efforts_untracked,
        parameters=params,
        policy=policy,
        title=title,
    )


# Backward-compatible name from the census-noise version.
def noisy_adaptive_trapper_effort(
    W_prev: float,
    p: Mapping[str, float],
    rng: np.random.Generator,
) -> float:
    """Deprecated alias: tracked policy plus one draw of untracked effort."""
    total, _, _ = total_hunting_effort(
        W_prev, p, include_untracked=True, rng=rng,
    )
    return total


__all__ = [
    "ADAPTIVE_TRAPPER_DEFAULTS",
    "SEASONAL_TRAPPER_DEFAULTS",
    "TrappingRun",
    "adaptive_trapper_effort",
    "make_seasonal_trapper_model",
    "noisy_adaptive_trapper_effort",
    "periodic_pulse",
    "solve_trapper_policy",
    "total_hunting_effort",
    "trapper_pressure",
    "untracked_hunting_effort",
    "wolf_elk_derivatives",
    "wolf_elk_trapper_rhs",
]
