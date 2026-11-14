"""Demo: seasonal trapper pulses on wolves only (no farmers, F=0).

Run:  python demo_seasonal_trappers.py

Saves ``seasonal_trappers_timeseries.png``. Policy knobs: T, pulse_width,
Tr_base, Tr_pulse, c_TW.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trapper_policies import make_seasonal_trapper_model, trapper_pressure


def pulse_windows(t_end: float, period: float, width: float) -> list[tuple[float, float]]:
    """Return (start, end) intervals for each open season overlapping [0, t_end]."""
    windows: list[tuple[float, float]] = []
    n = 0
    while n * period < t_end:
        start = n * period
        end = min(start + width, t_end)
        if end > start:
            windows.append((start, end))
        n += 1
    return windows


def main() -> None:
    model = make_seasonal_trapper_model()
    p = model.parameters
    y0 = (800.0, 40.0)
    t_span = (0.0, 120.0)
    solution = model.solve(y0, t_span=t_span, n_points=2401)

    fig, (ax_pop, ax_tr) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for start, end in pulse_windows(t_span[1], p["T"], p["pulse_width"]):
        ax_pop.axvspan(start, end, color="0.85", zorder=0)
        ax_tr.axvspan(start, end, color="0.85", zorder=0)

    ax_pop.plot(solution.t, solution.y[0], lw=2, label="Elk")
    ax_pop.plot(solution.t, solution.y[1], lw=2, label="Wolves")
    ax_pop.set(ylabel="Population", title=model.title)
    ax_pop.grid(alpha=0.25)
    ax_pop.legend(loc="best")

    tr_t = np.array([trapper_pressure(float(t), p) for t in solution.t])
    ax_tr.plot(solution.t, tr_t, color="tab:brown", lw=2, label="Tr(t)")
    ax_tr.set(xlabel="Time", ylabel="Trapper effort Tr(t)")
    ax_tr.grid(alpha=0.25)
    ax_tr.legend(loc="best")
    fig.tight_layout()

    out = Path(__file__).resolve().parent / "seasonal_trappers_timeseries.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"Initial wolves: {y0[1]:.1f}")
    print(f"Final wolves:   {solution.y[1, -1]:.3e}")
    print(f"Final elk:      {solution.y[0, -1]:.1f}")
    print(f"Farmers F={p['F']} (off in trapper scenarios)")
    print(
        f"Trapping seasons: every T={p['T']} for width={p['pulse_width']}, "
        f"Tr={p['Tr_base']} -> {p['Tr_base'] + p['Tr_pulse']}, c_TW={p['c_TW']}"
    )


if __name__ == "__main__":
    main()
