"""Generate frontier-introduction figures for their source folder and the paper."""
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


FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES = REPO / "VERYIMPORTANTPAPER" / "figures"
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

# USER-EDITABLE MODEL SETTINGS
# Change these values and rerun this file to regenerate the PNG and paper PDF figures.
FRONTIER_START = 10.0
FRONTIER_DURATION = 35.0
FRONTIER_END_TIME = 240.0
FRONTIER_TIME_POINTS = 1600
FRONTIER_INITIAL = (600.0, 50.0)  # (elk, wolves)
MAX_FARMERS = 40.0
MAX_HUNTERS = 25.0
EXPONENTIAL_RATE = 4.0
ARCTAN_STEEPNESS = 4.0
MODEL_PARAMETER_OVERRIDES = {"c_FW": 0.00009}  # e.g. {"c_FW": 0.0009}

# Settings for the separate Playground 2 phase-plane comparison.
PLAYGROUND2_START = 10.0
PLAYGROUND2_DURATION = 1.0
PLAYGROUND2_END_TIME = 240.0
PLAYGROUND2_TIME_POINTS = 800
PLAYGROUND2_INITIAL = (650.0, 110.0)  # (elk, wolves)
PLAYGROUND2_SHAPE = "smoothstep"
PLAYGROUND2_PARAMETER_OVERRIDES = {"c_FW": 0.00009}

# Plot settings
SCHEDULE_PLOT_END_TIME = 70.0
SCHEDULE_PLOT_POINTS = 500
SHAPES = ("linear", "exponential", "arctan")
COLORS = {"linear": "#2878B5", "exponential": "#E07A22", "arctan": "#3B9155"}


def ramp(time, shape: str, start: float, duration: float):
    """Normalized introduction schedule, clipped to its 0 and 1 endpoints."""
    u = np.clip((np.asarray(time, dtype=float) - start) / duration, 0.0, 1.0)
    if shape == "linear":
        value = u
    elif shape == "exponential":
        value = np.expm1(EXPONENTIAL_RATE * u) / np.expm1(EXPONENTIAL_RATE)
    elif shape == "arctan":
        steepness = ARCTAN_STEEPNESS
        value = (np.arctan(steepness * (2.0 * u - 1.0)) + np.arctan(steepness)) / (2.0 * np.arctan(steepness))
    elif shape == "smoothstep":
        value = 3.0 * u**2 - 2.0 * u**3
    else:
        raise ValueError(f"Unknown introduction shape: {shape}")
    return float(value) if np.ndim(value) == 0 else value


def make_system(shape: str) -> ODESystem:
    """Use ODESystem for integration while varying farmer/hunter pressure in time."""
    parameters = {**WOLF_ELK_DEFAULTS, **MODEL_PARAMETER_OVERRIDES}

    def rhs(time, state, p):
        level = ramp(time, shape, FRONTIER_START, FRONTIER_DURATION)
        local = {
            **p,
            "F": MAX_FARMERS * level,
            "H": MAX_HUNTERS * level,
        }
        return wolf_elk_rhs(time, state, local)

    return ODESystem(
        rhs=rhs,
        state_names=("Elk", "Wolves"),
        parameters=parameters,
        title=f"{shape.title()} frontier introduction",
    )


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    fig.savefig(PAPER_FIGURES / f"{name}.pdf", dpi=300)
    plt.close(fig)


def introduction_schedule_figure() -> None:
    times = np.linspace(0.0, SCHEDULE_PLOT_END_TIME, SCHEDULE_PLOT_POINTS)
    # ODESystem's plotting method is used for this derived time series too;
    # the plotted rows are the three normalized schedules evaluated above.
    plotting_model = ODESystem(
        rhs=lambda _t, state, _p: np.zeros_like(state),
        state_names=tuple(shape.title() for shape in SHAPES),
        title="Frontier introduction schedules",
    )
    solution = SimpleNamespace(
        t=times,
        y=np.vstack([ramp(times, shape, FRONTIER_START, FRONTIER_DURATION) for shape in SHAPES]),
    )
    fig, ax = plotting_model.plot_time_series(solution)
    fig.set_size_inches(5.0, 2.76)
    ax.set_title("")
    for line, shape in zip(ax.lines, SHAPES):
        line.set_color(COLORS[shape])
        line.set_linewidth(2.6)
    ax.axvline(FRONTIER_START, color="#666666", linestyle="--", linewidth=1.1)
    ax.axvline(FRONTIER_START + FRONTIER_DURATION, color="#666666", linestyle="--", linewidth=1.1)
    ax.set_ylabel("Fraction of maximum pressure")
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlim(0, SCHEDULE_PLOT_END_TIME)
    ax.legend()
    fig.tight_layout()
    save(fig, "frontier_introduction_schedules")


def population_response_figure() -> None:
    systems = {shape: make_system(shape) for shape in SHAPES}
    solutions = {
        shape: system.solve(
            FRONTIER_INITIAL, (0.0, FRONTIER_END_TIME), n_points=FRONTIER_TIME_POINTS
        )
        for shape, system in systems.items()
    }

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.34), sharex=True)
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
        ax.legend(title="Introduction")
    axes[0].set_title("Elk response")
    axes[1].set_title("Wolf response")
    axes[-1].set_xlabel("Time")
    for ax in axes:
        ax.axvline(FRONTIER_START, color="#666666", linestyle="--", linewidth=1.1)
        ax.axvline(FRONTIER_START + FRONTIER_DURATION, color="#666666", linestyle="--", linewidth=1.1)
        ax.legend(loc="best")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "frontier_introduction_population_response")


def playground2_phase_plane_figure() -> None:
    """Recreate playground2's wolf-versus-elk comparison with its settings."""
    initial = PLAYGROUND2_INITIAL
    start = PLAYGROUND2_START
    duration = PLAYGROUND2_DURATION
    parameter_values = {**WOLF_ELK_DEFAULTS, **PLAYGROUND2_PARAMETER_OVERRIDES}

    def gradual_rhs(time, state, p):
        level = ramp(time, PLAYGROUND2_SHAPE, start, duration)
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

    gradual_system = ODESystem(
        rhs=gradual_rhs, state_names=("Elk", "Wolves"),
        parameters=parameter_values, title="Gradual introduction",
    )
    full_force_system = ODESystem(
        rhs=full_force_rhs, state_names=("Elk", "Wolves"),
        parameters=parameter_values, title="Full force from start",
    )
    gradual = gradual_system.solve(
        initial, (0.0, PLAYGROUND2_END_TIME), n_points=PLAYGROUND2_TIME_POINTS
    )
    full_force = full_force_system.solve(
        initial, (0.0, PLAYGROUND2_END_TIME), n_points=PLAYGROUND2_TIME_POINTS
    )

    fig, ax = gradual_system.plot_phase_plane(
        [gradual, full_force],
        labels=("Gradual introductions", "Full force from start"),
        mark_start=False,
        direction_arrows=False,
    )
    fig.set_size_inches(5.0, 3.31)
    ax.set_title("")
    ax.lines[0].set_color("#2878B5")
    ax.lines[0].set_linewidth(2.3)
    ax.lines[1].set_color("#E07A22")
    ax.lines[1].set_linewidth(2.3)
    ax.lines[1].set_linestyle("--")
    ax.scatter(*initial, color="black", s=38, zorder=4, label="Initial population")
    ax.legend()
    fig.tight_layout()
    save(fig, "playground2_wolf_vs_elk")


if __name__ == "__main__":
    introduction_schedule_figure()
    population_response_figure()
    playground2_phase_plane_figure()
    print(f"Wrote figures to {FIGURES}")
