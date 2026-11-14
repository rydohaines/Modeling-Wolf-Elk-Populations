"""Reusable ODE simulation, analysis, and plotting for ecological systems.

Dependencies: numpy, scipy, matplotlib.

Use ``make_wolf_elk_model()`` for the included wolf–elk–human-pressure model,
or supply your own ``rhs(t, y, parameters)`` to :class:`ODESystem`.

Plots never call ``plt.show()``. Every plotting method returns ``(fig, ax)``
so figures can be customized or saved by a script or notebook.

Model assumptions: farmers and hunters are exogenous constant pressures, not
state variables. All defaults are illustrative, not empirically calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult, root


RHS = Callable[[float, NDArray[np.float64], Mapping[str, float]], ArrayLike]


@dataclass
class ODESystem:
    """A named ODE system with a configurable right-hand side.

    ``rhs(t, y, parameters)`` must return one derivative per state variable.
    Values in ``parameters`` are available to the RHS by key. Numerical
    integration uses ``scipy.integrate.solve_ivp``.
    """

    rhs: RHS
    state_names: Sequence[str]
    parameters: dict[str, float] = field(default_factory=dict)
    title: str = "ODE system"

    def __post_init__(self) -> None:
        self.state_names = tuple(self.state_names)
        if not self.state_names or len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state_names must be nonempty and unique")
        self.parameters = dict(self.parameters)

    @property
    def dimension(self) -> int:
        return len(self.state_names)

    def with_parameters(self, **updates: float) -> "ODESystem":
        """Return an independent copy of this model with updated parameters."""
        unknown = set(updates).difference(self.parameters)
        if unknown:
            raise KeyError(f"Unknown parameter(s): {sorted(unknown)}")
        return replace(self, parameters={**self.parameters, **updates})

    def evaluate(self, t: float, state: ArrayLike) -> NDArray[np.float64]:
        y = np.asarray(state, dtype=float)
        if y.shape != (self.dimension,):
            raise ValueError(f"Expected state shape ({self.dimension},), got {y.shape}")
        dydt = np.asarray(self.rhs(float(t), y, self.parameters), dtype=float)
        if dydt.shape != y.shape or not np.all(np.isfinite(dydt)):
            raise ValueError("RHS must return a finite derivative for each state")
        return dydt

    def solve(
        self,
        y0: ArrayLike,
        t_span: tuple[float, float] = (0, 100),
        *,
        n_points: int = 1001,
        t_eval: ArrayLike | None = None,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        **solver_kwargs,
    ) -> OptimizeResult:
        """Integrate an IVP and return a SciPy OptimizeResult (t and y arrays)."""
        initial = np.asarray(y0, dtype=float)
        if initial.shape != (self.dimension,) or not np.all(np.isfinite(initial)):
            raise ValueError(f"y0 must be a finite vector of length {self.dimension}")
        if t_span[1] <= t_span[0]:
            raise ValueError("t_span must have increasing start and end times")
        if t_eval is None:
            if n_points < 2:
                raise ValueError("n_points must be >= 2")
            times = np.linspace(t_span[0], t_span[1], n_points)
        else:
            times = np.asarray(t_eval, dtype=float)
            if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
                raise ValueError("t_eval must be a nonempty 1D array of finite times")
            if np.any(np.diff(times) <= 0) or times[0] < t_span[0] or times[-1] > t_span[1]:
                raise ValueError("t_eval must be increasing and inside t_span")

        result = solve_ivp(
            self.evaluate, t_span, initial, t_eval=times, method=method,
            rtol=rtol, atol=atol, **solver_kwargs,
        )
        if not result.success:
            raise RuntimeError(f"ODE integration failed: {result.message}")
        return result

    def solve_many(
        self, initial_conditions: Sequence[ArrayLike],
        t_span: tuple[float, float] = (0, 100), **kwargs,
    ) -> list[OptimizeResult]:
        """Solve the same model with several initial states."""
        return [self.solve(y0, t_span, **kwargs) for y0 in initial_conditions]

    def parameter_sweep(
        self, parameter: str, values: Sequence[float], y0: ArrayLike,
        t_span: tuple[float, float] = (0, 100), **solver_kwargs,
    ) -> dict[float, OptimizeResult]:
        """Simulate different values of one parameter without changing this model."""
        if parameter not in self.parameters:
            raise KeyError(f"Unknown parameter: {parameter}")
        return {
            float(value): self.with_parameters(**{parameter: float(value)}).solve(
                y0, t_span, **solver_kwargs
            )
            for value in values
        }

    @staticmethod
    def _axes(ax: Axes | None, figsize: tuple[float, float] = (8, 5)) -> tuple[Figure, Axes]:
        if ax is None:
            return plt.subplots(figsize=figsize)
        return ax.figure, ax

    def plot_time_series(
        self, solution: OptimizeResult, *, ax: Axes | None = None,
        indices: Sequence[int] | None = None, title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot selected population/state trajectories against time."""
        fig, ax = self._axes(ax)
        which = range(self.dimension) if indices is None else indices
        for i in which:
            ax.plot(solution.t, solution.y[i], lw=2, label=self.state_names[i])
        ax.set(xlabel="Time", ylabel="Population / state", title=title or self.title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_population_river(
        self, solution: OptimizeResult, *, ax: Axes | None = None,
        indices: Sequence[int] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot nonnegative populations as a stacked-area ('river') chart."""
        fig, ax = self._axes(ax)
        which = list(range(self.dimension)) if indices is None else list(indices)
        data = solution.y[which]
        if np.any(data < -1e-7):
            raise ValueError("Stacked population chart requires nonnegative states")
        ax.stackplot(solution.t, np.maximum(data, 0), labels=[self.state_names[i] for i in which], alpha=0.85)
        ax.set(xlabel="Time", ylabel="Combined population", title=f"{self.title}: population river")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax

    def plot_phase_plane(
        self, solutions: OptimizeResult | Sequence[OptimizeResult], *,
        x_index: int = 0, y_index: int = 1, ax: Axes | None = None,
        labels: Sequence[str] | None = None, mark_start: bool = True,
        mark_end: bool = False, direction_arrows: bool = True,
    ) -> tuple[Figure, Axes]:
        """Draw state-vs-state trajectories, optionally for several starts."""
        if x_index == y_index:
            raise ValueError("Choose different phase-plane variables")
        fig, ax = self._axes(ax)
        items = [solutions] if isinstance(solutions, OptimizeResult) else list(solutions)
        if labels is not None and len(labels) != len(items):
            raise ValueError("labels must match the number of solutions")
        for k, sol in enumerate(items):
            x, y = sol.y[x_index], sol.y[y_index]
            label = labels[k] if labels is not None else f"Initial state {k + 1}"
            (line,) = ax.plot(x, y, lw=2, label=label)
            if mark_start:
                ax.scatter(x[0], y[0], marker="o", s=32, color=line.get_color(), zorder=4)
            if mark_end:
                ax.scatter(x[-1], y[-1], marker="x", s=55, color=line.get_color(), zorder=4)
            if direction_arrows and len(x) >= 4:
                for fraction in (0.25, 0.65):
                    j = min(int(fraction * (len(x) - 1)), len(x) - 2)
                    ax.annotate("", (x[j + 1], y[j + 1]), (x[j], y[j]),
                                arrowprops={"arrowstyle": "->", "color": line.get_color(), "lw": 1.7})
        ax.set(xlabel=self.state_names[x_index], ylabel=self.state_names[y_index],
               title=f"{self.title}: phase plane")
        ax.grid(alpha=0.25)
        if len(items) > 1 or labels is not None:
            ax.legend()
        fig.tight_layout()
        return fig, ax

    def vector_field(
        self, xlim: tuple[float, float], ylim: tuple[float, float], *,
        x_index: int = 0, y_index: int = 1, grid: int = 25,
        t: float = 0.0, fixed_state: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the (possibly sliced) 2D vector field on a regular grid.

        For higher-dimensional systems provide ``fixed_state`` to hold other
        states constant while varying the two selected axes.
        """
        if x_index == y_index or not (0 <= x_index < self.dimension and 0 <= y_index < self.dimension):
            raise ValueError("x_index and y_index must be different valid state indices")
        if grid < 3 or xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
            raise ValueError("grid must be >= 3 and axis limits must be increasing")
        if fixed_state is None:
            if self.dimension != 2:
                raise ValueError("For systems with >2 states, pass fixed_state to define the slice")
            state = np.zeros(self.dimension)
        else:
            state = np.asarray(fixed_state, dtype=float)
            if state.shape != (self.dimension,):
                raise ValueError("fixed_state must have one entry per state variable")
        x = np.linspace(*xlim, grid)
        y = np.linspace(*ylim, grid)
        X, Y = np.meshgrid(x, y)
        U = np.empty_like(X)
        V = np.empty_like(Y)
        for row in range(grid):
            for col in range(grid):
                current = state.copy()
                current[x_index] = X[row, col]
                current[y_index] = Y[row, col]
                derivative = self.evaluate(t, current)
                U[row, col] = derivative[x_index]
                V[row, col] = derivative[y_index]
        return X, Y, U, V

    def plot_quiver(
        self, xlim: tuple[float, float], ylim: tuple[float, float], *,
        ax: Axes | None = None, grid: int = 21, normalize: bool = True,
        x_index: int = 0, y_index: int = 1, t: float = 0,
        fixed_state: ArrayLike | None = None,
    ) -> tuple[Figure, Axes]:
        """Arrow plot (direction field). Normalization emphasizes direction."""
        X, Y, U, V = self.vector_field(xlim, ylim, x_index=x_index, y_index=y_index,
                                        grid=grid, t=t, fixed_state=fixed_state)
        if normalize:
            speed = np.hypot(U, V)
            U = np.divide(U, speed, out=np.zeros_like(U), where=speed > 0)
            V = np.divide(V, speed, out=np.zeros_like(V), where=speed > 0)
        fig, ax = self._axes(ax)
        ax.quiver(X, Y, U, V, angles="xy", pivot="mid")
        ax.set(xlim=xlim, ylim=ylim, xlabel=self.state_names[x_index],
               ylabel=self.state_names[y_index], title=f"{self.title}: quiver field")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig, ax

    def plot_streamplot(
        self, xlim: tuple[float, float], ylim: tuple[float, float], *,
        ax: Axes | None = None, grid: int = 51, density: float = 1.2,
        x_index: int = 0, y_index: int = 1, t: float = 0,
        fixed_state: ArrayLike | None = None,
    ) -> tuple[Figure, Axes]:
        """Streamline ('river') plot of the 2D flow, colored by log speed."""
        X, Y, U, V = self.vector_field(xlim, ylim, x_index=x_index, y_index=y_index,
                                        grid=grid, t=t, fixed_state=fixed_state)
        fig, ax = self._axes(ax)
        speed = np.log1p(np.hypot(U, V))
        ax.streamplot(X[0], Y[:, 0], U, V, density=density, color=speed,
                      cmap="viridis", linewidth=1.2, arrowsize=1.0)
        ax.set(xlim=xlim, ylim=ylim, xlabel=self.state_names[x_index],
               ylabel=self.state_names[y_index], title=f"{self.title}: stream/river field")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        return fig, ax

    # An easy-to-find alias for users who call streamplots 'river graphs'.
    plot_river = plot_streamplot

    def plot_nullclines(
        self, xlim: tuple[float, float], ylim: tuple[float, float], *,
        ax: Axes | None = None, grid: int = 121,
        x_index: int = 0, y_index: int = 1, t: float = 0,
        fixed_state: ArrayLike | None = None,
    ) -> tuple[Figure, Axes]:
        """Show where each phase-plane derivative vanishes (dX=0, dY=0)."""
        X, Y, U, V = self.vector_field(xlim, ylim, x_index=x_index, y_index=y_index,
                                        grid=grid, t=t, fixed_state=fixed_state)
        fig, ax = self._axes(ax)
        ax.contour(X, Y, U, levels=[0], colors=["tab:blue"], linewidths=2)
        ax.contour(X, Y, V, levels=[0], colors=["tab:orange"], linewidths=2)
        ax.legend(handles=[
            Line2D([0], [0], color="tab:blue", lw=2, label=f"d({self.state_names[x_index]})/dt = 0"),
            Line2D([0], [0], color="tab:orange", lw=2, label=f"d({self.state_names[y_index]})/dt = 0"),
        ])
        ax.set(xlim=xlim, ylim=ylim, xlabel=self.state_names[x_index],
               ylabel=self.state_names[y_index], title=f"{self.title}: nullclines")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig, ax

    def jacobian(self, equilibrium: ArrayLike, *, t: float = 0.0,
                 relative_step: float = 1e-6) -> NDArray[np.float64]:
        """Numerically approximate the Jacobian using central differences."""
        x = np.asarray(equilibrium, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError("equilibrium must have one value per state variable")
        J = np.empty((self.dimension, self.dimension))
        for k in range(self.dimension):
            h = relative_step * max(1.0, abs(x[k]))
            dx = np.zeros(self.dimension)
            dx[k] = h
            J[:, k] = (self.evaluate(t, x + dx) - self.evaluate(t, x - dx)) / (2 * h)
        return J

    def stability(self, equilibrium: ArrayLike, *, tol: float = 1e-7) -> dict:
        """Linearization-based local stability classification at a fixed point.

        Meaningful only if the supplied point is an equilibrium.
        Eigenvalues with near-zero real parts are inconclusive.
        """
        point = np.asarray(equilibrium, dtype=float)
        residual = np.linalg.norm(self.evaluate(0.0, point))
        J = self.jacobian(point)
        eig = np.linalg.eigvals(J)
        re = eig.real
        if np.all(re < -tol):
            label = "locally asymptotically stable"
        elif np.any(re > tol) and np.any(re < -tol):
            label = "unstable (saddle-type)"
        elif np.any(re > tol):
            label = "unstable"
        else:
            label = "inconclusive (non-hyperbolic)"
        return {"point": point, "residual": residual, "jacobian": J,
                "eigenvalues": eig, "classification": label}

    def find_equilibria(
        self, xlim: tuple[float, float], ylim: tuple[float, float], *,
        starts_per_axis: int = 7, residual_tol: float = 1e-7,
        dedup_tol: float = 1e-3,
    ) -> list[NDArray[np.float64]]:
        """Numerically find distinct 2D equilibria in a bounding rectangle.

        Multi-start root finding is heuristic: it may miss solutions. Requires
        an autonomous, two-state model (RHS independent of time).
        """
        if self.dimension != 2:
            raise ValueError("find_equilibria currently supports only 2D models")
        if starts_per_axis < 2 or xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
            raise ValueError("Need >= 2 starts per axis and valid limits")
        equilibria: list[NDArray[np.float64]] = []

        def safe_residual(z: NDArray[np.float64]) -> NDArray[np.float64]:
            # Root solvers may try arbitrarily large iterates. A finite penalty
            # lets that trial fail gracefully instead of aborting the search.
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.asarray(self.rhs(0.0, np.asarray(z), self.parameters), dtype=float)
            if values.shape != (2,):
                raise ValueError("RHS must return two derivatives")
            if not np.all(np.isfinite(values)):
                return np.full(2, 1e30)
            return values

        for x in np.linspace(*xlim, starts_per_axis):
            for y in np.linspace(*ylim, starts_per_axis):
                candidate = root(safe_residual, [x, y])
                if not candidate.success or not np.all(np.isfinite(candidate.x)):
                    continue
                p = np.asarray(candidate.x, dtype=float)
                # Root finding can represent a boundary equilibrium such as
                # E = 0 as a tiny signed floating-point residual. Snap only
                # values already within the deduplication tolerance, then
                # verify the projected point still satisfies the ODE.
                p[np.abs(p) < dedup_tol] = 0.0
                if np.linalg.norm(self.evaluate(0.0, p)) > residual_tol:
                    continue
                if not (xlim[0] - dedup_tol <= p[0] <= xlim[1] + dedup_tol and
                        ylim[0] - dedup_tol <= p[1] <= ylim[1] + dedup_tol):
                    continue
                if not any(np.linalg.norm(p - old) < dedup_tol for old in equilibria):
                    equilibria.append(p)
        return sorted(equilibria, key=lambda z: (z[0], z[1]))


def periodic_pulse(
    t: float,
    period: float,
    width: float,
    height: float,
    base: float = 0.0,
) -> float:
    """Rectangular periodic pulse: base most of the time, base+height for width each period.

    Within each interval ``[n*period, (n+1)*period)``, the pulse is active on
    ``[n*period, n*period + width)``. Requires ``period > 0`` and ``0 <= width <= period``.
    """
    if period <= 0:
        raise ValueError("period must be strictly positive")
    if width < 0 or width > period:
        raise ValueError("width must satisfy 0 <= width <= period")
    phase = float(t) % period
    if phase < width:
        return base + height
    return base


def farmer_pressure(t: float, p: Mapping[str, float]) -> float:
    """Return exogenous farmer pressure F, possibly time-varying via pulse params.

    If ``F_base``, ``F_pulse``, ``T``, and ``pulse_width`` are all present, use
    ``periodic_pulse``. Otherwise use constant ``F``.
    """
    pulse_keys = ("F_base", "F_pulse", "T", "pulse_width")
    if all(key in p for key in pulse_keys):
        return periodic_pulse(
            t, period=p["T"], width=p["pulse_width"],
            height=p["F_pulse"], base=p["F_base"],
        )
    return float(p["F"])


def wolf_elk_rhs(t: float, y: NDArray[np.float64], p: Mapping[str, float]) -> NDArray[np.float64]:
    """Wolf–elk dynamics with farming and hunting pressure.

    dE/dt = r_E*E*(1-E/(K-alpha_FE*F)) - a*E*W - c_HE*H*E
    dW/dt = eta*a*E*W - d_W*W - c_FW*F*W

    E = elk; W = wolves; F = farmer population/effort; H = hunter effort.
    H is a fixed external input. F may be constant or a continuous periodic
    pulse when pulse parameters are supplied (see ``farmer_pressure``).
    This is a simplified mechanistic example, not a validated field model.
    """
    E, W = y
    F = farmer_pressure(t, p)
    effective_capacity = p["K"] - p["alpha_FE"] * F
    if effective_capacity <= 0:
        raise ValueError("Farmer pressure must leave a positive elk carrying capacity")
    elk_growth = p["r_E"] * E * (1 - E / effective_capacity)
    predation = p["a"] * E * W
    hunter_elk_loss = p["c_HE"] * p["H"] * E
    farmer_wolf_loss = p["c_FW"] * F * W
    dE = elk_growth - predation - hunter_elk_loss
    dW = p["eta"] * predation - p["d_W"] * W - farmer_wolf_loss
    return np.array([dE, dW], dtype=float)


WOLF_ELK_DEFAULTS: dict[str, float] = {
    "r_E": 0.40,    # intrinsic elk per-capita growth, 1/time
    "K": 1200.0,    # elk carrying capacity without farmer pressure, elk
    "alpha_FE": 10.0, # capacity reduction per farmer, elk/farmer
    "a": 0.001,     # predation encounters, 1/(wolf*time)
    "eta": 0.25,    # conversion of elk consumed into wolf recruitment (scaled)
    "d_W": 0.08,    # natural wolf per-capita death, 1/time
    "F": 6.0,       # farmer population / pressure index, exogenous
    "H": 4.0,       # hunter population / pressure index, exogenous
    "c_HE": 0.003,  # elk loss per hunter, 1/(hunter*time)
    "c_FW": 0.004,  # wolf loss per farmer, 1/(farmer*time)
}

# Continuous periodic farmer pulses. Defaults are illustrative and chosen so
# repeated spikes drive wolves toward extinction over several periods.
PULSED_FARMER_DEFAULTS: dict[str, float] = {
    **WOLF_ELK_DEFAULTS,
    "F_base": 2.0,       # baseline farmer pressure between pulses
    "F_pulse": 90.0,     # added farmer pressure during each pulse window
    "T": 15.0,           # time between pulse starts
    "pulse_width": 4.0,  # duration of each high-pressure window
    "c_FW": 0.02,        # stronger wolf loss per farmer so pulses bite hard
    "F": 2.0,            # kept in sync with F_base for capacity checks / API
}


def _validate_wolf_elk_parameters(parameters: Mapping[str, float], *, max_F: float) -> None:
    if parameters["K"] <= 0:
        raise ValueError("K must be strictly positive")
    if parameters["K"] - parameters["alpha_FE"] * max_F <= 0:
        raise ValueError("Farmer pressure must leave a positive elk carrying capacity")
    if any(v < 0 for v in parameters.values()):
        raise ValueError("Wolf–elk parameters must be nonnegative")


def make_wolf_elk_model(**parameter_overrides: float) -> ODESystem:
    """Construct the example model; override parameters using keyword args.

    Example: ``make_wolf_elk_model(F=10, H=0, K=1500)``.
    """
    invalid = set(parameter_overrides) - set(WOLF_ELK_DEFAULTS)
    if invalid:
        raise KeyError(f"Unknown wolf–elk parameter(s): {sorted(invalid)}")
    parameters = {**WOLF_ELK_DEFAULTS, **parameter_overrides}
    _validate_wolf_elk_parameters(parameters, max_F=parameters["F"])
    return ODESystem(
        rhs=wolf_elk_rhs, state_names=("Elk", "Wolves"),
        parameters=parameters, title="Wolf–elk ecosystem",
    )


def make_pulsed_farmer_model(**parameter_overrides: float) -> ODESystem:
    """Wolf–elk model with continuous periodic farmer-pressure pulses F(t).

    Between pulses F = F_base; during each window of length ``pulse_width``
    starting at t = 0, T, 2T, ... farmers jump to F_base + F_pulse. Only
    farmers remove wolves; hunters remain a constant elk harvest.

    Example: ``make_pulsed_farmer_model(T=15, F_pulse=90, pulse_width=2)``.
    """
    invalid = set(parameter_overrides) - set(PULSED_FARMER_DEFAULTS)
    if invalid:
        raise KeyError(f"Unknown pulsed-farmer parameter(s): {sorted(invalid)}")
    parameters = {**PULSED_FARMER_DEFAULTS, **parameter_overrides}
    # Keep scalar F aligned with baseline for any code that reads it directly.
    if "F_base" in parameter_overrides and "F" not in parameter_overrides:
        parameters["F"] = parameters["F_base"]
    elif "F" in parameter_overrides and "F_base" not in parameter_overrides:
        parameters["F_base"] = parameters["F"]
    period = parameters["T"]
    width = parameters["pulse_width"]
    if period <= 0:
        raise ValueError("T must be strictly positive")
    if width < 0 or width > period:
        raise ValueError("pulse_width must satisfy 0 <= pulse_width <= T")
    max_F = parameters["F_base"] + parameters["F_pulse"]
    _validate_wolf_elk_parameters(parameters, max_F=max_F)
    return ODESystem(
        rhs=wolf_elk_rhs, state_names=("Elk", "Wolves"),
        parameters=parameters, title="Wolf–elk ecosystem (pulsed farmers)",
    )


__all__ = [
    "ODESystem",
    "WOLF_ELK_DEFAULTS",
    "PULSED_FARMER_DEFAULTS",
    "farmer_pressure",
    "make_pulsed_farmer_model",
    "make_wolf_elk_model",
    "periodic_pulse",
    "wolf_elk_rhs",
]
