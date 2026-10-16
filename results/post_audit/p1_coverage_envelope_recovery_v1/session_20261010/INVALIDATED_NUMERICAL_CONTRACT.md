# Invalidated pilot: 2026-10-10

The run was interrupted after a code review discovered an unintended float32
radial sampling scale. `torch.where(mask, math.sqrt(2.), 1.)` produces float32
scales although the samples and evaluated density are float64. The resulting
scale is 1.4142135381698608 rather than float64 sqrt(2). This is not an exact
match to the declared Gaussian sampling law.

Preserve the frozen source archive, reservations and completed checkpoint units.
No completed prefix, mean, precision, or selected seed from this run qualifies.
The interrupted unit remains reserved and is not replaced. A new whole protocol
namespace and run directory, with a float64-scaling regression test, are required.
No statistical failure is removed; this is a numerical-contract invalidation.
