# G11 V9 Development V1 Execution Failure

Date: 2026-08-02

Status: **namespace invalid for inference; no result artifact was produced.**

The frozen V1 development executor stopped atomically after 222.9 seconds because
one external-comparator allocation pilot contained 3 nonzero inferential units,
below the frozen minimum of 4.  The exception occurred before aggregation and
before the output file was written.  No V1 performance result exists.

The nonzero-support criterion is not lowered.  V2 retains every scientific gate,
cell, method, path budget, cluster count, and target RMSE, but increases independent
pilot units from 64 to 256 and uses a fresh seed namespace.  This is a resource and
allocation correction, not an outcome-dependent performance change.  V1 seeds may
not be reused or described as confirmation data.
