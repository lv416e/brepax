# Event-Time Sensitivity and the Limits of TOI Correction

Method B is unimplemented. The valid calculation below differentiates an
event time; it does **not** derive an exact correction to a static Boolean
volume gradient. Earlier versions conflated these objects and gave
incompatible signs for the proposed gradient and VJP.

## Boundary Functions

For disks with positive radii and center distance $d=\|c_1-c_2\|$:

$$
g_{\mathrm{ext}}(\theta)=d-r_1-r_2,\qquad
g_{\mathrm{int}}(\theta)=|r_1-r_2|-d.
$$

Partial overlap satisfies both guards strictly negative. Equal radii and
coincident centers can make the internal guard nonsmooth and need separate
treatment. Ordinary disk area and sphere volume are $C^1$ at both external
and unequal-radius internal tangency; see
[Stratification Theory](stratification_theory.md#contact-order-for-area-and-volume).

## Derivative of a Simple Event Time

Fix a direction $v$ and path $\theta(t)=\theta_0+t v$ in the valid parameter
domain. Suppose a smooth guard $g$ has a finite isolated root $\tau>0$,
the selected root remains the same local event, and

$$
\nabla g(\theta_\tau)^\top v\ne0,\qquad \theta_\tau=\theta_0+\tau v.
$$

The implicit function theorem applied to $g(\theta_0+\tau v)=0$ gives

$$
D_{\theta_0}\tau
=-\frac{\nabla g(\theta_\tau)^\top}{\nabla g(\theta_\tau)^\top v}.
$$

This differentiates the event time with respect to initial parameters,
holding $v$ fixed. If $v$ depends on $\theta_0$, its derivative also enters.
Selecting the first of multiple roots introduces event-order changes not
covered by this local formula.

A zero denominator means transversality fails, not that no event exists.
For example, $g(t)=\sqrt{4+(t-1)^2}-2$ has finite contact at $t=1$ with
$g'(1)=0$. Clamping the denominator or returning zero defines a fallback,
not a justified derivative.

## Why This Is Not a Static Gradient Correction

A future event on a chosen path does not alter the local derivative of
$f(\theta_0)$. An exact volume locally independent of a center coordinate
has zero derivative in that direction.

The earlier rule added
$[(\nabla f_j-\nabla f_i)^\top v]D_{\theta_0}\tau$ to $\nabla f_i$.
Take $f(x)=\max(0,x)$, $x_0=-1$, $v=1$, and guard $g(x)=x$.
Then $\tau=1$, $D_{x_0}\tau=-1$, and the derivative jump is $1$.
The proposed rule returns $-1$ instead of the true static derivative $0$.
The former VJP used the opposite sign and returned $+1$, also incorrect.

Event-time sensitivity can enter a specified look-ahead objective or a
trajectory/update-map derivative. A deliberately nonlocal optimization
signal is another possibility. Each needs its own contract and validation;
none follows from adding a future derivative jump to a static gradient.

## Contact Dynamics and JAX

Hybrid dynamics differentiates a flow with guards and reset maps. A
saltation matrix describes sensitivity across a transition. Static Boolean
volume has no such flow/reset map by default. Sharing an implicit-function
calculation is an analogy, not a mathematical isomorphism. See
[Kong et al., Saltation Matrices](https://arxiv.org/abs/2306.06862).

A JVP must be linear in its **tangent input**, while its coefficients can
depend nonlinearly on primal values. Residual dependence alone does not
require `jax.custom_vjp`. See
[JAX custom derivative rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).

## Implementation Status

`brepax.boolean.toi` has no implementation; `method="toi"` raises
`NotImplementedError`. [ADR-0009](../architecture/adr/0009-method-b-deferred.md)
retains the deferral. A future method must first specify whether it
differentiates static geometry, an event time, or a trajectory, including
grazing and simultaneous events.
