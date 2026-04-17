"""From STEP file to DFM gradient in 10 lines of code.

Demonstrates the end-to-end pipeline: load a STEP file, reconstruct
a differentiable CSG representation, compute multiple DFM metrics,
and obtain gradients of each metric w.r.t. design parameters.

This is BRepAX's core value proposition: connecting CAD geometry
to gradient-based manufacturing analysis.
"""

import equinox as eqx
import jax.numpy as jnp

from brepax.brep.csg_stump import reconstruct_csg_stump, stump_to_differentiable
from brepax.io.step import read_step
from brepax.metrics import (
    draft_angle_violation,
    surface_area,
    thin_wall_volume,
)


def main() -> None:
    # --- Pipeline: STEP file to differentiable shape ---
    shape = read_step("tests/fixtures/box_with_holes.step")
    stump = reconstruct_csg_stump(shape)
    diff = stump_to_differentiable(stump)

    # Evaluation domain (slightly larger than the part)
    lo = jnp.array([-1.0, -1.0, -1.0])
    hi = jnp.array([41.0, 31.0, 21.0])
    res = 32

    # --- Compute DFM metrics ---
    print("=== DFM Metrics ===\n")

    vol = diff.volume(resolution=res, lo=lo, hi=hi)
    print(f"Volume:              {float(vol):10.1f} mm^3")

    area = surface_area(diff.sdf, lo=lo, hi=hi, resolution=res)
    print(f"Surface area:        {float(area):10.1f} mm^2")

    min_wall = 2.0
    thin = thin_wall_volume(diff.sdf, min_wall, lo=lo, hi=hi, resolution=res)
    print(f"Thin wall (t<{min_wall}mm):  {float(thin):10.1f} mm^3")

    pull_dir = jnp.array([0.0, 0.0, 1.0])
    min_draft = jnp.radians(5.0)
    draft = draft_angle_violation(
        diff.sdf, pull_dir, min_draft, lo=lo, hi=hi, resolution=res
    )
    print(f"Draft violation (5deg): {float(draft):8.1f} mm^2")

    # --- Gradients w.r.t. design parameters ---
    print("\n=== Gradients (resolution=16 for speed) ===\n")

    lo16, hi16, res16 = lo, hi, 16

    grad_vol = eqx.filter_grad(lambda d: d.volume(resolution=res16, lo=lo16, hi=hi16))(
        diff
    )

    grad_area = eqx.filter_grad(
        lambda d: surface_area(d.sdf, lo=lo16, hi=hi16, resolution=res16)
    )(diff)

    grad_thin = eqx.filter_grad(
        lambda d: thin_wall_volume(d.sdf, min_wall, lo=lo16, hi=hi16, resolution=res16)
    )(diff)

    grad_draft = eqx.filter_grad(
        lambda d: draft_angle_violation(
            d.sdf, pull_dir, min_draft, lo=lo16, hi=hi16, resolution=res16
        )
    )(diff)

    # Display gradient for each primitive's key parameter
    print(
        f"{'Primitive':20s} | {'d(Vol)':>10s} | {'d(Area)':>10s} | {'d(Thin)':>10s} | {'d(Draft)':>10s}"
    )
    print("-" * 75)

    for i, (p, gv, ga, gt, gd) in enumerate(
        zip(
            diff.primitives,
            grad_vol.primitives,
            grad_area.primitives,
            grad_thin.primitives,
            grad_draft.primitives,
            strict=True,
        )
    ):
        params = p.parameters()
        grad_params_v = gv.parameters()
        grad_params_a = ga.parameters()
        grad_params_t = gt.parameters()
        grad_params_d = gd.parameters()

        # Pick the most informative gradient (radius for cylinders, offset for planes)
        if "radius" in params:
            key = "radius"
        elif "half_extents" in params:
            key = "half_extents"
        elif "offset" in params:
            key = "offset"
        else:
            continue

        dv = grad_params_v.get(key)
        da = grad_params_a.get(key)
        dt = grad_params_t.get(key)
        dd = grad_params_d.get(key)

        if dv is None:
            continue

        label = f"{type(p).__name__}[{i}].{key}"

        # Summarize as scalar (norm for vectors)
        def scalar(x):
            if x.ndim == 0:
                return float(x)
            return float(jnp.linalg.norm(x))

        print(
            f"{label:20s} | {scalar(dv):10.2f} | {scalar(da):10.2f} "
            f"| {scalar(dt):10.2f} | {scalar(dd):10.2f}"
        )

    print("\n--- Physical interpretation ---")
    print("d(Volume)/d(hole_radius) < 0: enlarging a hole reduces volume")
    print("d(Surface area)/d(hole_radius) > 0: enlarging a hole adds surface")
    print("d(Thin wall)/d(hole_radius) > 0: enlarging a hole increases thin-wall risk")
    print(
        "d(Draft)/d(pull_direction): rotating pull direction changes draft violations"
    )


if __name__ == "__main__":
    main()
