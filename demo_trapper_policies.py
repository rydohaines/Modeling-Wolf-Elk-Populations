"""Compare tracked wolf hunting vs the same policy plus untracked harvest.

Run:  python demo_trapper_policies.py

The adaptive rule uses the true lagged wolf census (no census noise). Untracked
runs add random extra wolf removals on top of that policy. Populations are
normalized by initial values. Saves ``trapper_policies_comparison.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from trapper_policies import solve_trapper_policy


def plot_normalized(ax, run, y0, *, title: str) -> None:
    e0, w0 = y0
    target = run.parameters["W_target"]
    ax.plot(run.t, run.y[0] / e0, lw=2, label="Elk / E0")
    ax.plot(run.t, run.y[1] / w0, lw=2, label="Wolves / W0")
    ax.axhline(target / w0, color="0.4", ls="--", lw=1, label=f"W_target / W0 = {target / w0:.2f}")
    ax.set(ylabel="Relative population", title=title)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def main() -> None:
    y0 = (800.0, 40.0)
    t_span = (0.0, 400.0)

    tracked = solve_trapper_policy(y0, t_span, policy="adaptive", dt=0.05)
    with_untracked_persist = solve_trapper_policy(
        y0, t_span, policy="untracked", dt=0.05, seed=1,
    )
    with_untracked_crash = solve_trapper_policy(
        y0, t_span, policy="untracked", dt=0.05, seed=0,
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    plot_normalized(
        axes[0], tracked, y0,
        title="Tracked adaptive wolf hunting only: stable stock",
    )
    plot_normalized(
        axes[1], with_untracked_persist, y0,
        title="Same policy + untracked hunting (seed=1): wolves persist",
    )
    plot_normalized(
        axes[2], with_untracked_crash, y0,
        title="Same policy + untracked hunting (seed=0): wolves crash",
    )
    axes[2].set_xlabel("Time")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "trapper_policies_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    p = tracked.parameters
    print(
        f"F={p['F']}, Tr_untracked={p['Tr_untracked']}, "
        f"sigma_untracked={p['sigma_untracked']}"
    )
    print(
        f"Tracked final wolves: {tracked.y[1, -1]:.2f} "
        f"({tracked.y[1, -1] / y0[1]:.2f}×W0) | "
        f"Untracked seed1: {with_untracked_persist.y[1, -1]:.2f} "
        f"({with_untracked_persist.y[1, -1] / y0[1]:.2f}×W0) | "
        f"Untracked seed0: {with_untracked_crash.y[1, -1]:.2f}"
    )


if __name__ == "__main__":
    main()
