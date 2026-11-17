# Wolf–elk ODE analysis

## Terms and notation

| Symbol or term | Meaning |
|---|---|
| \(t\) | Time. |
| \(E(t)\) | Elk population at time \(t\). |
| \(W(t)\) | Wolf population at time \(t\). |
| \(F\) | Constant farmer pressure or effort. |
| \(H\) | Constant hunter pressure or effort. |
| \(r_E\) | Elk intrinsic per-capita growth rate. |
| \(K\) | Elk carrying capacity before farmer effects. |
| \(\alpha_{FE}\) | Reduction in elk carrying capacity per unit farmer pressure. |
| \(C=K-\alpha_{FE}F\) | Effective elk carrying capacity under farmer pressure. |
| \(a\) | Wolf predation rate coefficient for elk. |
| \(c_{HE}\) | Elk loss rate per unit hunter pressure. |
| \(\eta\) | Conversion factor from elk consumed to wolf recruitment. |
| \(d_W\) | Natural wolf per-capita death rate. |
| \(c_{FW}\) | Wolf loss rate per unit farmer pressure. |
| \(m=r_E-c_{HE}H\) | Elk per-capita growth rate at low elk density, with hunters present. |
| \(q=d_W+c_{FW}F\) | Wolf per-capita loss rate, with farmer pressure present. |
| \(\dot E,\dot W\) | Rates of change of the elk and wolf populations. |
| Equilibrium | A population state where both rates of change are zero. |
| \((E_0,0)\) | Elk-only equilibrium; \(E_0\) is its elk population. |
| \((E_*,W_*)\) | Coexistence equilibrium; \(E_*\) and \(W_*\) are its elk and wolf populations. |
| \(J(E,W)\) | Jacobian matrix of the ODE system at \((E,W)\). |

This note analyzes the basic, constant-pressure model in `wolf_elk_demo.ipynb`. Farmer pressure `F` and hunter pressure `H` are fixed inputs, not dynamic populations. The seasonal and pulsed variants are outside this analysis.

## Model

Let `E(t) >= 0` be elk and `W(t) >= 0` be wolves. The notebook uses

\[
\begin{aligned}
\dot E &= r_E E\left(1-\frac{E}{K-\alpha_{FE}F}\right)-aEW-c_{HE}HE,\\
\dot W &= \eta aEW-d_WW-c_{FW}FW.
\end{aligned}
\]

For constant `F`, define the effective elk carrying capacity, elk net growth rate at low density, and wolf per-capita loss rate as

\[
C=K-\alpha_{FE}F,\qquad m=r_E-c_{HE}H,\qquad q=d_W+c_{FW}F.
\]

The model requires `C > 0`; the supplied model builder also requires parameters nonnegative. In this notation,

\[
\dot E=E\left(m-\frac{r_E}{C}E-aW\right),\qquad
\dot W=W(\eta aE-q).
\]

Assume below that `r_E > 0`, `a > 0`, and `eta > 0`. The formulas simplify or become degenerate if one of these is zero.

## Equilibria and local stability

Stability is for the continuous-time ODE within the nonnegative (`E,W`) quadrant. The Jacobian is

\[
J(E,W)=\begin{pmatrix}
m-2r_EE/C-aW & -aE\\
\eta aW & \eta aE-q
\end{pmatrix}.
\]

| Equilibrium | Existence in the nonnegative quadrant | Stability conditions |
|---|---|---|
| Extinction \((0,0)\) | Always | Locally asymptotically stable if \(m \leq 0\) and \(q > 0\); stable but not asymptotically stable if \(m \leq 0\) and \(q = 0\); unstable if \(m > 0\). |
| Elk-only \((E_0,0)\), where \(E_0 = \dfrac{Cm}{r_E}\) | \(m > 0\) | Locally asymptotically stable if \(q \geq \eta a E_0\) (nonhyperbolic at equality); saddle if \(q < \eta a E_0\). |
| Coexistence \((E_*,W_*)\) | \(E_* = \dfrac{q}{\eta a}\) and \(W_* = \dfrac{m-r_E E_*/C}{a}\); strictly positive if \(q > 0\) and \(m > \dfrac{r_Eq}{\eta a C}\) | Locally asymptotically stable if \(q > 0\) and \(m > \dfrac{r_Eq}{\eta a C}\). |

The coexistence condition can also be written as

\[
\eta a C m > r_E q.
\]

At the coexistence point, the Jacobian has

\[
\operatorname{tr}J=-\frac{r_EE_*}{C}<0,\qquad
\det J=\eta a^2E_*W_*>0,
\]

These signs give local asymptotic stability.

## Notebook default parameters

The basic model defaults are

| Parameter | Value |
|---|---:|
| `r_E` | 0.40 |
| `K` | 1200 |
| `alpha_FE` | 10 |
| `a` | 0.001 |
| `eta` | 0.25 |
| `d_W` | 0.08 |
| `F` | 6 |
| `H` | 4 |
| `c_HE` | 0.003 |
| `c_FW` | 0.004 |

These give `C=1140`, `m=0.388`, and `q=0.104`. The equilibria are approximately:

- Extinction: `(0, 0)`, unstable because its elk eigenvalue is `0.388 > 0`.
- Elk-only: `(1105.8, 0)`, a saddle because wolves have positive invasion rate `eta*a*E_0-q ~= 0.1724`.
- Coexistence: `(416, 242)`, locally asymptotically stable.

These values are illustrative, not calibrated ecological estimates.
