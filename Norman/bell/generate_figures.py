"""Generate Beddington--DeAngelis replacements for the paper figures."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NORMAN = HERE.parent
REPO = NORMAN.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(NORMAN))

from ecosystem_ode import ODESystem
from repo_model_snapshot import WOLF_ELK_DEFAULTS, wolf_elk_rhs


OUTPUT_DIR = HERE

# Shared Beddington--DeAngelis model settings for the flow and introduction plots.
MODEL_PARAMETER_OVERRIDES = {"c_FW": 0.00009}
FRONTIER_START = 10.0
FRONTIER_DURATION = 35.0
FRONTIER_END_TIME = 240.0
FRONTIER_TIME_POINTS = 1600
FRONTIER_INITIAL = (600.0, 50.0)
MAX_FARMERS = 40.0
MAX_HUNTERS = 25.0
PLAYGROUND2_INITIAL = (650.0, 110.0)
PLAYGROUND2_DURATION = 1.0

ELK_RANGE = (0.0, 1250.0)
WOLF_RANGE = (0.0, 300.0)
STREAM_GRID = 55
STREAM_DENSITY = 1.15
SCHEDULE_END_TIME = 70.0
SCHEDULE_POINTS = 500
SHAPES = ("linear", "exponential", "arctan")
COLORS = {"linear": "#2878B5", "exponential": "#E07A22", "arctan": "#3B9155"}


def ramp(time, shape: str, start: float, duration: float):
    """Normalized introduction schedule clipped to zero and one."""
    u = np.clip((np.asarray(time, dtype=float) - start) / duration, 0.0, 1.0)
    if shape == "linear":
        value = u
    elif shape == "exponential":
        value = np.expm1(4.0 * u) / np.expm1(4.0)
    elif shape == "arctan":
        value = (np.arctan(4.0 * (2.0 * u - 1.0)) + np.arctan(4.0)) / (2.0 * np.arctan(4.0))
    elif shape == "smoothstep":
        value = 3.0 * u**2 - 2.0 * u**3
    else:
        raise ValueError(f"Unknown introduction shape: {shape}")
    return float(value) if np.ndim(value) == 0 else value


def model(parameters=None, title="Beddington--DeAngelis wolf--elk model"):
    return ODESystem(
        rhs=wolf_elk_rhs,
        state_names=("Elk", "Wolves"),
        parameters={**WOLF_ELK_DEFAULTS, **MODEL_PARAMETER_OVERRIDES, **(parameters or {})},
        title=title,
    )


def save(fig, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def simple_model_flow() -> None:
    """Save the constant-pressure Beddington--DeAngelis vector field."""
    # Match the paper's Figure 1 policy settings: F=6, H=4, c_FW=0.004.
    system = model(
        {"c_FW": WOLF_ELK_DEFAULTS["c_FW"]},
        title="Beddington--DeAngelis wolf--elk phase-plane flow",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    system.plot_streamplot(
        ELK_RANGE, WOLF_RANGE, ax=ax, grid=STREAM_GRID, density=STREAM_DENSITY
    )
    ax.set_title("Beddington--DeAngelis wolf--elk phase-plane flow")
    fig.tight_layout()
    save(fig, "simple_model_flow.png")


def introduction_schedules() -> None:
    times = np.linspace(0.0, SCHEDULE_END_TIME, SCHEDULE_POINTS)
    plot_model = ODESystem(
        rhs=lambda _t, state, _p: np.zeros_like(state),
        state_names=tuple(shape.title() for shape in SHAPES),
        title="Beddington--DeAngelis model: farmer and hunter schedules",
    )
    solution = SimpleNamespace(
        t=times,
        y=np.vstack([ramp(times, shape, FRONTIER_START, FRONTIER_DURATION) for shape in SHAPES]),
    )
    fig, ax = plot_model.plot_time_series(solution)
    for line, shape in zip(ax.lines, SHAPES):
        line.set_color(COLORS[shape])
        line.set_linewidth(2.6)
    ax.axvline(FRONTIER_START, color="#666666", linestyle="--", linewidth=1.1)
    ax.axvline(FRONTIER_START + FRONTIER_DURATION, color="#666666", linestyle="--", linewidth=1.1)
    ax.set_ylabel("Fraction of maximum farmer and hunter pressure")
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlim(0.0, SCHEDULE_END_TIME)
    ax.set_title("Farmer and hunter introductions in the Beddington--DeAngelis model")
    ax.legend()
    fig.tight_layout()
    save(fig, "frontier_introduction_schedules.png")


def introduction_population_response() -> None:
    parameters = {**WOLF_ELK_DEFAULTS, **MODEL_PARAMETER_OVERRIDES}
    systems = {}
    solutions = {}
    for shape in SHAPES:
        def rhs(time, state, p, shape=shape):
            level = ramp(time, shape, FRONTIER_START, FRONTIER_DURATION)
            return wolf_elk_rhs(time, state, {
                **p,
                "F": MAX_FARMERS * level,
                "H": MAX_HUNTERS * level,
            })

        systems[shape] = ODESystem(
            rhs=rhs, state_names=("Elk", "Wolves"), parameters=parameters,
            title=f"{shape.title()} farmer and hunter introduction",
        )
        solutions[shape] = systems[shape].solve(
            FRONTIER_INITIAL, (0.0, FRONTIER_END_TIME), n_points=FRONTIER_TIME_POINTS
        )

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.0), sharex=True)
    for shape in SHAPES:
        solution = solutions[shape]
        for index, ax in enumerate(axes):
            systems[shape].plot_time_series(solution, ax=ax, indices=(index,))
            line = ax.lines[-1]
            line.set_color(COLORS[shape])
            line.set_linewidth(2.1)
            line.set_label(shape.title())
    for ax, species in zip(axes, ("Elk", "Wolves")):
        ax.set_ylabel(f"{species} population")
        ax.grid(alpha=0.22)
        ax.axvline(FRONTIER_START, color="#666666", linestyle="--", linewidth=1.1)
        ax.axvline(FRONTIER_START + FRONTIER_DURATION, color="#666666", linestyle="--", linewidth=1.1)
        ax.legend(title="Introduction function", loc="best")
    axes[0].set_title("Elk response")
    axes[1].set_title("Wolf response")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Beddington--DeAngelis population response to frontier introductions", y=0.99)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    save(fig, "frontier_introduction_population_response.png")


def gradual_vs_full_force() -> None:
    parameters = {**WOLF_ELK_DEFAULTS, **MODEL_PARAMETER_OVERRIDES}

    def gradual_rhs(time, state, p):
        level = ramp(time, "smoothstep", FRONTIER_START, PLAYGROUND2_DURATION)
        return wolf_elk_rhs(time, state, {
            **p,
            "F": MAX_FARMERS * level,
            "H": MAX_HUNTERS * level,
        })

    def full_force_rhs(time, state, p):
        return wolf_elk_rhs(time, state, {
            **p,
            "F": MAX_FARMERS,
            "H": MAX_HUNTERS,
        })

    gradual = ODESystem(
        rhs=gradual_rhs, state_names=("Elk", "Wolves"), parameters=parameters,
        title="Gradual introduction",
    )
    full = ODESystem(
        rhs=full_force_rhs, state_names=("Elk", "Wolves"), parameters=parameters,
        title="Full force from start",
    )
    gradual_solution = gradual.solve(PLAYGROUND2_INITIAL, (0.0, FRONTIER_END_TIME), n_points=800)
    full_solution = full.solve(PLAYGROUND2_INITIAL, (0.0, FRONTIER_END_TIME), n_points=800)
    fig, ax = gradual.plot_phase_plane(
        [gradual_solution, full_solution],
        labels=("Gradual introductions", "Full force from start"),
        mark_start=False,
        direction_arrows=False,
    )
    ax.lines[0].set_color(COLORS["linear"])
    ax.lines[0].set_linewidth(2.3)
    ax.lines[1].set_color(COLORS["exponential"])
    ax.lines[1].set_linewidth(2.3)
    ax.lines[1].set_linestyle("--")
    ax.scatter(*PLAYGROUND2_INITIAL, color="black", s=38, zorder=4, label="Initial population")
    ax.set_title("Beddington--DeAngelis model: gradual pressure vs. full force")
    ax.legend()
    fig.tight_layout()
    save(fig, "playground2_wolf_vs_elk.png")


if __name__ == "__main__":
    simple_model_flow()
    introduction_schedules()
    introduction_population_response()
    gradual_vs_full_force()
    print(f"Wrote Beddington--DeAngelis replacement figures to {OUTPUT_DIR}")
