# TOI Correction Derivation

Mathematical derivation of the time-of-impact gradient correction for Method (B).

## Setup

Design parameter vector $\theta = (c_{1x}, c_{1y}, r_1, c_{2x}, c_{2y}, r_2) \in \mathbb{R}^6$.

Center distance $d(\theta) = \|c_1 - c_2\|$.

Union area $f(\theta)$ is continuous but has discontinuous gradient across stratum boundaries.

## 1. Boundary Functions and Sign Convention

Two boundary surfaces partition parameter space into three strata:

**External tangent** (disjoint $\leftrightarrow$ intersecting):

$$g_{\text{ext}}(\theta) = d(\theta) - (r_1 + r_2)$$

- $g_{\text{ext}} > 0$: disjoint
- $g_{\text{ext}} < 0$: overlapping (intersecting or contained)
- $g_{\text{ext}} = 0$: external tangent

**Internal tangent** (intersecting $\leftrightarrow$ contained):

$$g_{\text{int}}(\theta) = |r_1 - r_2| - d(\theta)$$

- $g_{\text{int}} > 0$: contained
- $g_{\text{int}} < 0$: not contained (intersecting or disjoint)
- $g_{\text{int}} = 0$: internal tangent

Convention: **positive $g$ means the boundary has been crossed away from the intersecting stratum**. The intersecting stratum is the region where both $g_{\text{ext}} < 0$ and $g_{\text{int}} < 0$.

## 2. TOI Quantity

Given current parameters $\theta_0$ and a perturbation direction $v \in \mathbb{R}^6$, the time-of-impact for boundary $g$ is:

$$\tau(\theta_0, v) = \min\{t > 0 : g(\theta_0 + t \cdot v) = 0\}$$

**Domain: $\tau > 0$ only.** We compute the first forward crossing along the perturbation direction. If no crossing exists ($g(\theta_0 + t \cdot v) \neq 0$ for all $t > 0$), then $\tau = +\infty$ and no correction is needed.

For the disk-disk case, $g_{\text{ext}}(\theta_0 + tv) = d(\theta_0 + tv) - (r_1 + t \cdot v_{r_1}) - (r_2 + t \cdot v_{r_2})$ is generally nonlinear in $t$ (through the norm $d$), so $\tau$ is found by root-finding rather than closed-form. In practice, `optimistix.bisection` or Newton iteration suffices.

### Implicit Function Theorem for $\partial\tau/\partial\theta_0$

Define $h(t, \theta_0) = g(\theta_0 + t \cdot v)$. At $t = \tau$, $h(\tau, \theta_0) = 0$.

By the IFT, provided $\partial h/\partial t \neq 0$:

$$\frac{\partial \tau}{\partial \theta_0} = -\frac{\partial h / \partial \theta_0}{\partial h / \partial t} \bigg|_{t=\tau}$$

Computing the partials:

$$\frac{\partial h}{\partial t}\bigg|_{t=\tau} = \nabla g(\theta_\tau)^\top v$$

$$\frac{\partial h}{\partial \theta_0}\bigg|_{t=\tau} = \nabla g(\theta_\tau)^\top$$

where $\theta_\tau = \theta_0 + \tau \cdot v$. Therefore:

$$\frac{\partial \tau}{\partial \theta_0} = -\frac{\nabla g(\theta_\tau)^\top}{\nabla g(\theta_\tau)^\top v}$$

This is a row vector in $\mathbb{R}^{1 \times 6}$, linear in $\nabla g$.

**Linearity note:** $\partial\tau/\partial\theta_0$ is a ratio of linear functions of $\nabla g$, hence it is **not** affine in $\theta_0$ in general. This means `jax.custom_jvp` is insufficient; **`jax.custom_vjp` is required** because the correction depends on the residual $\theta_\tau$ which is itself a function of $\theta_0$.

## 3. Degenerate Cases

The IFT requires $\nabla g(\theta_\tau)^\top v \neq 0$, i.e., the perturbation must not be tangent to the boundary at the crossing point.

**Concentric disks** ($c_1 = c_2$, so $d = 0$): $\nabla_{c} d = (c_1 - c_2)/d$ is undefined. Handle by detecting $d < \epsilon_{\text{safe}}$ and falling back to the analytical gradient for the contained stratum (which is well-defined at $d = 0$: $\partial f/\partial r_1 = 2\pi r_1$).

**Equal radii** ($r_1 = r_2$): $|r_1 - r_2| = 0$, so the internal tangent boundary passes through $d = 0$, which is the concentric case above. For $d > 0$ with $r_1 = r_2$, $g_{\text{int}} = -d < 0$ always, so the internal boundary is never reached and no correction is needed.

**Tangent perturbation** ($\nabla g^\top v = 0$): The perturbation slides along the boundary without crossing. No TOI exists ($\tau = +\infty$), no correction needed.

Implementation: guard with `jnp.where(jnp.abs(denom) > eps_safe, correction, 0.0)`.

## 4. Gradient Correction Formula

The union area $f(\theta)$ is continuous across boundaries but $\nabla f$ is discontinuous. For $\theta_0$ in stratum $S_i$ with area formula $f_i$, and adjacent stratum $S_j$ across boundary $g = 0$:

**Naive gradient** (within-stratum autodiff):

$$\nabla f_{\text{naive}}(\theta_0) = \nabla f_i(\theta_0)$$

**TOI-corrected gradient:**

$$\nabla f_{\text{corrected}}(\theta_0) = \nabla f_i(\theta_0) + \Delta \nabla f \cdot \frac{\partial \tau}{\partial \theta_0} \cdot (\nabla g(\theta_\tau)^\top v)$$

Wait -- let me state this more carefully. The correction arises when we want the gradient of "what would happen if we crossed the boundary." The total derivative of $f$ along a path that crosses the boundary at $\theta_\tau$:

$$\frac{df(\theta_0 + tv)}{dt}\bigg|_{t=\tau^+} - \frac{df(\theta_0 + tv)}{dt}\bigg|_{t=\tau^-} = [\nabla f_j(\theta_\tau) - \nabla f_i(\theta_\tau)]^\top v$$

The boundary crossing time $\tau$ depends on $\theta_0$, so the chain rule gives the correction:

$$\nabla f_{\text{corrected}} = \nabla f_i(\theta_0) + \underbrace{[\nabla f_j(\theta_\tau) - \nabla f_i(\theta_\tau)]^\top v}_{\text{gradient jump} \times \text{direction}} \cdot \underbrace{\frac{\partial \tau}{\partial \theta_0}}_{\text{boundary sensitivity}}$$

Substituting $\partial\tau/\partial\theta_0$:

$$\nabla f_{\text{corrected}} = \nabla f_i(\theta_0) - \frac{[\nabla f_j(\theta_\tau) - \nabla f_i(\theta_\tau)]^\top v}{\nabla g(\theta_\tau)^\top v} \cdot \nabla g(\theta_\tau)^\top$$

Note that the $v$ terms cancel in the ratio when the gradient jump is parallel to $\nabla g$, leaving a purely geometric correction.

**In the VJP formulation** (which is what JAX uses), the correction to the cotangent vector $\bar{f}$ is:

$$\bar{\theta}_{\text{corrected}} = \bar{\theta}_{\text{naive}} + \bar{f} \cdot \frac{[\nabla f_j(\theta_\tau) - \nabla f_i(\theta_\tau)]^\top v}{\nabla g(\theta_\tau)^\top v} \cdot \nabla g(\theta_\tau)$$

Since $\bar{f}$ is scalar (area is scalar output), this simplifies the implementation.

## 5. Contact Dynamics Correspondence

The TOI correction is structurally isomorphic to rigid-body contact handling in differentiable physics:

| Physics simulation | BRepAX |
|---|---|
| State $q$ (position, velocity) | Design parameters $\theta$ |
| Time step direction $\dot{q} \cdot \Delta t$ | Perturbation direction $v$ |
| Contact surface $\phi(q) = 0$ | Stratum boundary $g(\theta) = 0$ |
| Time of impact $t_c$ | Boundary crossing $\tau$ |
| Contact normal $\nabla\phi / \|\nabla\phi\|$ | Boundary normal $\nabla g / \|\nabla g\|$ |
| Velocity impulse $\Delta v$ | Gradient jump $\nabla f_j - \nabla f_i$ |
| Post-contact state sensitivity $\partial q^+/\partial q^-$ | Corrected gradient $\nabla f_{\text{corrected}}$ |

The mathematical structure is identical: both involve differentiating through a root-finding problem (IFT) and applying a correction at the event boundary. The "impulse" in physics (discontinuous velocity change) corresponds to the gradient jump (discontinuous $\nabla f$) in BRepAX. The contact normal determines the direction of the impulse, just as $\nabla g$ determines the direction of the gradient correction.

This is not a metaphor but a **mathematical isomorphism**: the same IFT formula applies to both domains, with the substitution table above.

## 6. Disk-Disk Concrete Example

### Boundary gradients

Unit direction from $c_1$ to $c_2$: $\hat{e} = (c_2 - c_1) / d$.

**External tangent** $g_{\text{ext}} = d - r_1 - r_2$:

$$\nabla g_{\text{ext}} = \begin{pmatrix} -\hat{e}_x \\ -\hat{e}_y \\ -1 \\ \hat{e}_x \\ \hat{e}_y \\ -1 \end{pmatrix}$$

(moving $c_1$ toward $c_2$ decreases $d$, decreasing $g_{\text{ext}}$; increasing $r_1$ decreases $g_{\text{ext}}$)

**Internal tangent** $g_{\text{int}} = |r_1 - r_2| - d$ (assuming $r_1 > r_2$):

$$\nabla g_{\text{int}} = \begin{pmatrix} \hat{e}_x \\ \hat{e}_y \\ 1 \\ -\hat{e}_x \\ -\hat{e}_y \\ -1 \end{pmatrix}$$

### TOI for the external tangent

For $\theta_0$ in the intersecting stratum ($g_{\text{ext}} < 0$), perturbing along $v$:

$$\tau_{\text{ext}} = \frac{-g_{\text{ext}}(\theta_0)}{\nabla g_{\text{ext}}(\theta_0)^\top v} + O(\tau^2)$$

The linear approximation suffices when $\theta_0$ is close to the boundary ($|g_{\text{ext}}|$ small). For the exact $\tau$, root-find $g_{\text{ext}}(\theta_0 + tv) = 0$.

### Correction term at external tangent

At the boundary ($d = r_1 + r_2$), the gradient jump is:

$$\Delta\nabla f = \nabla f_{\text{disjoint}}(\theta_\tau) - \nabla f_{\text{intersecting}}(\theta_\tau)$$

For $f_{\text{disjoint}} = \pi r_1^2 + \pi r_2^2$ (no overlap):

$$\nabla f_{\text{disjoint}} = (0, 0, 2\pi r_1, 0, 0, 2\pi r_2)^\top$$

For $f_{\text{intersecting}}$, the gradient is the analytical formula involving $\alpha$, $\beta$ (see `disk_disk_union_area`), evaluated at the tangent configuration.

At external tangency ($\alpha = 0$, $\beta = 0$): $f_{\text{intersecting}}$ gradient w.r.t. $r_1$ is $2\pi r_1$, and gradient w.r.t. center components is $0$. So $\Delta\nabla f = 0$ at the tangent point itself.

This means the TOI correction vanishes at exact tangency for the area -- because the area function's gradient is actually continuous at external tangency (the intersection area formula reduces to the disjoint formula as overlap goes to zero). **The gradient discontinuity at external tangency is second-order, not first-order.**

This is a significant finding: for the disk-disk union area, the TOI correction at external tangent is zero to first order. The correction becomes non-trivial only for configurations where the gradient jump is genuinely discontinuous, which occurs at **internal tangency** (containment transition), where $\nabla f_{\text{contained}} \neq \nabla f_{\text{intersecting}}$ in the center components.

### Correction term at internal tangent

At internal tangency ($d = |r_1 - r_2|$, assuming $r_1 > r_2$):

$f_{\text{contained}} = \pi r_1^2$, so $\nabla f_{\text{contained}} = (0, 0, 2\pi r_1, 0, 0, 0)^\top$.

$f_{\text{intersecting}}$ evaluated at $d = r_1 - r_2$ gives gradients involving the $\alpha, \beta$ terms of the lens formula, with non-zero components in the center directions ($\partial f / \partial c_{1x} \neq 0$).

Therefore $\Delta\nabla f \neq 0$ at internal tangency, and the TOI correction is non-trivial.

## Implementation Plan

1. Forward pass: compute `union_area` using standard branchless `jnp.where` (same as Method A without smoothing, i.e., exact SDF evaluation on grid)
2. `jax.custom_vjp` wrapper:
   - `fwd`: save residuals `(primal_out, stratum_label, g_ext, g_int, theta)`
   - `bwd`: compute naive gradient + TOI correction using the formulas above
3. Root-finding for $\tau$: use linear approximation (first-order TOI) for simplicity; upgrade to `optimistix.bisection` if accuracy demands
4. Degenerate guards: `safe_d`, denominator clamping as in the analytical function
