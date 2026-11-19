"""Wolf Extinction Model (Adam): elk response when wolves are gone.

Run:  python demo_wolf_extinction.py

Saves:
  wolf_extinction_timeseries.png  — policy-driven crash vs W=0 from t=0
  wolf_extinction_comparison.png  — with wolves vs without (elk + vegetation)
  wolf_extinction_elk_boom.png    — final elk vs KF under W≡0

The two figures used in VERYIMPORTANTPAPER are also written as PDFs to
VERYIMPORTANTPAPER/figures at their final 5-inch publication width.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecosystem_ode import make_wolf_elk_model

OUT = Path(__file__).resolve().parent
T_SPAN = (0.0, 80.0)
N = 1601
PAPER_FIGURES = OUT / "VERYIMPORTANTPAPER" / "figures"

# Policy-driven extinction run. Exposed at module level so other work
# (e.g. wolf_reintroduction.ipynb) can start from where this run ends.
EXTINCTION_PARAMS = {"KF": 110.0, "hE": 0.04, "hW": 0.35, "pF": 0.45}
EXTINCTION_Y0 = (45.0, 12.0)


def vegetation(E: np.ndarray, KF: float) -> np.ndarray:
    return np.clip(1.0 - E / max(KF, 1e-6), 0.0, 1.0)


def main() -> None:
    with_wolves = make_wolf_elk_model(KF=100.0, hE=0.01, hW=0.0, pF=0.0)
    extinction_policy = make_wolf_elk_model(**EXTINCTION_PARAMS)
    no_wolves = make_wolf_elk_model(KF=110.0, hE=0.04, hW=0.0, pF=0.0)
    # Same habitat/hunting as coexistence, but wolves removed — fair vegetation compare.
    no_wolves_matched = make_wolf_elk_model(KF=100.0, hE=0.01, hW=0.0, pF=0.0)

    y0_coexist = (55.0, 18.0)
    y0_pressure = EXTINCTION_Y0
    y0_gone = (45.0, 0.0)

    sol_coexist = with_wolves.solve(y0_coexist, T_SPAN, n_points=N)
    sol_crash = extinction_policy.solve(y0_pressure, T_SPAN, n_points=N)
    sol_gone = no_wolves.solve(y0_gone, T_SPAN, n_points=N)
    sol_gone_matched = no_wolves_matched.solve((55.0, 0.0), T_SPAN, n_points=N)

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.06), sharex=True)
    axes[0].plot(sol_crash.t, sol_crash.y[0], lw=2, label="Elk")
    axes[0].plot(sol_crash.t, sol_crash.y[1], lw=2, label="Wolves")
    axes[0].set(
        ylabel="Population",
        title="Policy-driven wolf extinction",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(sol_gone.t, sol_gone.y[0], lw=2, color="tab:blue", label="Elk (W=0)")
    axes[1].axhline(
        no_wolves.parameters["KF"] * max(0.0, 1.0 - no_wolves.parameters["hE"] / no_wolves.parameters["r"]),
        color="0.35", ls="--", lw=1.2,
        label="Elk carrying eq. under hunting",
    )
    axes[1].set(
        xlabel="Time",
        ylabel="Population",
        title="Elk growth after wolf extinction",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    path = OUT / "wolf_extinction_timeseries.png"
    fig.savefig(path, dpi=300)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "wolf_extinction_timeseries.pdf", dpi=300)
    print(f"Saved {path}")

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.06), sharex=True)
    axes[0].plot(sol_coexist.t, sol_coexist.y[0], lw=2, label="Elk (with wolves)")
    axes[0].plot(sol_gone_matched.t, sol_gone_matched.y[0], lw=2, ls="--", label="Elk (no wolves)")
    axes[0].plot(sol_coexist.t, sol_coexist.y[1], lw=2, label="Wolves")
    axes[0].set(ylabel="Population", title="Populations under matched habitat and elk hunting")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    veg_with = vegetation(sol_coexist.y[0], with_wolves.parameters["KF"])
    veg_gone = vegetation(sol_gone_matched.y[0], no_wolves_matched.parameters["KF"])
    axes[1].plot(sol_coexist.t, veg_with, lw=2, label="Vegetation proxy (with wolves)")
    axes[1].plot(sol_gone_matched.t, veg_gone, lw=2, ls="--", label="Vegetation proxy (no wolves)")
    axes[1].set(
        xlabel="Time",
        ylabel="Relative vegetation (1 − E/KF)",
        title="Vegetation proxy",
    )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    path = OUT / "wolf_extinction_comparison.png"
    fig.savefig(path, dpi=300)
    fig.savefig(PAPER_FIGURES / "wolf_extinction_comparison.pdf", dpi=300)
    print(f"Saved {path}")

    KFs = np.linspace(40.0, 120.0, 17)
    finals = []
    for KF in KFs:
        model = make_wolf_elk_model(KF=float(KF), hE=0.04, hW=0.0, pF=0.0)
        sol = model.solve((45.0, 0.0), (0.0, 100.0), n_points=801)
        finals.append(sol.y[0, -1])
    finals = np.asarray(finals)
    analytic = KFs * np.maximum(0.0, 1.0 - 0.04 / 0.55)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(KFs, finals, "o-", lw=2, label="Simulated final elk (W=0)")
    ax.plot(KFs, analytic, "--", lw=1.8, color="0.35", label=r"$E^*=K_F(1-h_E/r)$")
    ax.set(
        xlabel="Agricultural carrying capacity KF",
        ylabel="Long-run elk population",
        title="No-wolf elk boom vs habitat capacity (fixed elk hunting hE=0.04)",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = OUT / "wolf_extinction_elk_boom.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    print(
        f"Coexist final: E={sol_coexist.y[0, -1]:.1f}, W={sol_coexist.y[1, -1]:.2f} | "
        f"Crash final: E={sol_crash.y[0, -1]:.1f}, W={sol_crash.y[1, -1]:.3e} | "
        f"Gone final: E={sol_gone.y[0, -1]:.1f}"
    )


if __name__ == "__main__":
    main()
