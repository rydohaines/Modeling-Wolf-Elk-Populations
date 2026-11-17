"""Small fallback copy of the repository's constant-pressure wolf–elk model."""
import numpy as np


WOLF_ELK_DEFAULTS = {
    "r_E": 0.40,
    "K": 1200.0,
    "alpha_FE": 10.0,
    "a": 0.001,
    "eta": 0.25,
    "d_W": 0.08,
    "F": 6.0,
    "H": 4.0,
    "c_HE": 0.003,
    "c_FW": 0.004,
}


def wolf_elk_rhs(t, y, p):
    """Repository model equations for state (elk, wolves)."""
    elk, wolves = y
    farmers = float(p["F"])
    effective_capacity = p["K"] - p["alpha_FE"] * farmers
    if effective_capacity <= 0:
        raise ValueError("Farmer pressure must leave positive elk capacity")
    elk_growth = p["r_E"] * elk * (1 - elk / effective_capacity)
    predation = p["a"] * elk * wolves
    hunter_elk_loss = p["c_HE"] * p["H"] * elk
    farmer_wolf_loss = p["c_FW"] * farmers * wolves
    elk_change = elk_growth - predation - hunter_elk_loss
    wolf_change = p["eta"] * predation - p["d_W"] * wolves - farmer_wolf_loss
    return np.array([elk_change, wolf_change], dtype=float)
