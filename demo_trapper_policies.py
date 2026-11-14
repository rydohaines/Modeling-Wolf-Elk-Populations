"""Compare noiseless vs noisy census trapping (no farmers).

Run:  python demo_trapper_policies.py

The adaptive rule is stable without noise. The identical rule with census noise
can eliminate wolves on some seeds. Saves ``trapper_policies_comparison.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from trapper_policies import solve_trapper_policy


def main() -> None:
    y0 = (800.0, 40.0)
    t_span = (0.0, 400.0)

    adaptive = solve_trapper_policy(y0, t_span, policy="adaptive", dt=0.05)
    # Same defaults; only the RNG seed changes between persist vs crash paths.
    noisy_persist = solve_trapper_policy(
        y0, t_span, policy="noisy_adaptive", dt=0.05, seed=10,
    )
    noisy_crash = solve_trapper_policy(
        y0, t_span, policy="noisy_adaptive", dt=0.05, seed=3,
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    target = adaptive.parameters["W_target"]

    axes[0].plot(adaptive.t, adaptive.y[0], lw=2, label="Elk")
    axes[0].plot(adaptive.t, adaptive.y[1], lw=2, label="Wolves")
    axes[0].axhline(target, color="0.4", ls="--", lw=1, label=f"W_target={target:g}")
    axes[0].set(
        ylabel="Population",
        title="Adaptive (no noise, F=0): stable positive wolf stock",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(noisy_persist.t, noisy_persist.y[0], lw=2, label="Elk")
    axes[1].plot(noisy_persist.t, noisy_persist.y[1], lw=2, label="Wolves")
    axes[1].axhline(target, color="0.4", ls="--", lw=1, label=f"W_target={target:g}")
    axes[1].set(
        ylabel="Population",
        title="Same policy + census noise (seed=10): wolves persist",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(noisy_crash.t, noisy_crash.y[0], lw=2, label="Elk")
    axes[2].plot(noisy_crash.t, noisy_crash.y[1], lw=2, label="Wolves")
    axes[2].axhline(target, color="0.4", ls="--", lw=1, label=f"W_target={target:g}")
    axes[2].set(
        xlabel="Time",
        ylabel="Population",
        title="Same policy + census noise (seed=3): wolves crash",
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "trapper_policies_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"F={adaptive.parameters['F']} (farmers off)")
    print(
        f"Adaptive final wolves: {adaptive.y[1, -1]:.2f} | "
        f"Noisy seed10: {noisy_persist.y[1, -1]:.2f} | "
        f"Noisy seed3: {noisy_crash.y[1, -1]:.2f}"
    )


if __name__ == "__main__":
    main()
