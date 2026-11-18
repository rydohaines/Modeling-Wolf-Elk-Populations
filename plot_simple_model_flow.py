"""Plot the simple wolf–elk phase-plane flow for one parameter set.

Edit ``PARAMETER_VARIATIONS`` below, then run:

    python plot_simple_model_flow.py

The phase-plane streamplot has elk on x and wolves on y. By default the
figure is displayed and saved as ``simple_model_flow.png``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from ecosystem_ode import ODESystem
from Norman.repo_model_snapshot import WOLF_ELK_DEFAULTS, wolf_elk_rhs


# Set parameter overrides here. Unlisted parameters use the model defaults
# from Norman/repo_model_snapshot.py. For example: {"F": 10.0, "H": 6.0}
PARAMETER_OVERRIDES = {}

# Plot settings. Adjust these ranges to zoom in or out on the phase plane.
ELK_RANGE = (0, 1250)
WOLF_RANGE = (0, 300)
GRID_SIZE = 55
STREAM_DENSITY = 1.15
OUTPUT_FILE = "simple_model_flow.png"


def main() -> None:
    parameters = {**WOLF_ELK_DEFAULTS, **PARAMETER_OVERRIDES}
    model = ODESystem(
        rhs=wolf_elk_rhs,
        state_names=("Elk", "Wolves"),
        parameters=parameters,
        title="Simple wolf–elk model",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    model.plot_streamplot(
        ELK_RANGE,
        WOLF_RANGE,
        ax=ax,
        grid=GRID_SIZE,
        density=STREAM_DENSITY,
    )
    ax.set_title("Wolf–elk phase-plane flow")
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
