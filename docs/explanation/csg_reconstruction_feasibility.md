# CSG Reconstruction Scope

BRepAX reconstructs candidate CSG representations from supported B-Rep faces.
This is implemented functionality, with limits that differ from the original
proposal on this page. Reconstruction does not recover a unique modeling
history or guarantee an exact representation of every input solid.

## Implemented paths

| Path | Behavior | Scope |
|---|---|---|
| `reconstruct_stock_minus_features` | Recognizes a box stock and groups connected cylindrical feature faces using the adjacency graph. | A restricted recognition heuristic, not a general feature recognizer; candidate output needs independent validation. |
| `reconstruct_csg_stump` | Samples primitive sign vectors, classifies representative points with OCCT point membership, and builds a union of intersection terms (DNF). | Sampling can miss cells. Unsupported faces cause reconstruction to return `None`; successful reconstruction is not an accuracy certificate. |
| `group_stump_primitives` and `compact_stump` | Group recognized bounded primitives and simplify intersection terms. | These operations do not establish equivalence to the source CAD or recover its original feature history. |

Adjacency is used for feature grouping. The sampled DNF reconstruction does
not implement the earlier proposal's adjacency-pruned search over a learned
Boolean connection matrix.

## Differentiation boundary

OCCT extraction, point-membership classification, and CSG reconstruction run
on the host. `DifferentiableCSGStump` exposes primitive parameters to JAX, but
its intersection matrix and union mask are static. Its derivatives therefore
do not differentiate the reconstruction process or optimize connection choices.

`DifferentiableCSGStump.volume` integrates a sigmoid indicator on a grid and
has discretization and smoothing error. It does not use analytical stratum
dispatch. Separate Boolean APIs, including selected cases in
`evaluate_stump_volume_stratum`, use stratum-dispatched measures; see the
[API overview](../reference/index.md#boolean-measures-and-trimmed-surfaces)
for their limits. A finite gradient or a small optimization loss does not by
itself establish an accurate CAD volume or shape derivative.

Trimmed-face distances cannot simply replace half-space ingredients in a DNF.
The attempted trim-aware composition failed on analytical and B-spline cases;
[ADR-0020](../architecture/adr/0020-marschner-bspline-csg-scope.md) records the
counterexample and current raw-primitive-SDF rule. Face-level trimming remains
useful for standalone distance queries, but is not a demonstrated fallback for
general solid reconstruction.

## Evidence and further development

The [integration report](../reference/benchmarks.md) records measurements on
selected fixtures. It makes no claim that BRepAX is the first or only system
combining B-Rep input, topology, and differentiable evaluation. The previous
assertions on this page that no prior work used this combination were not
supported by a systematic literature comparison and are withdrawn.

Useful extensions should be evaluated against concrete reconstruction and
editing tasks: compare point membership and measures with the source CAD,
check gradients against independent derivatives, and retain failures and
unsupported inputs. Vary sampling and discretization before attributing an
improvement to the reconstruction method. Claims about topology-changing edits
need separate evidence beyond derivatives of a fixed CSG representation.

The earlier effort estimates, predicted success rates, and proposed learned
connection matrix were planning assumptions, not measured results or the
current implementation contract. A future novelty or comparative-performance
claim would require a current source-backed literature review and a comparable
evaluation; this page establishes neither.
