# G11 V8 R2i reference-proposal manifest

Date: 2026-07-31  
Gate: complete method-cell proposal freeze  
Decision: pass for a new execution configuration

The manifest contains exactly 48 unique entries: 24 threshold-bound cells times two
independent reference methods.

- 37 entries retain the original threshold-calibration proposal because their
  formal pilot allocations already fit the resource cap.
- 9 failed entries use the audited all-failed-cells V3 selections.
- 1 barrier raw entry uses the audited dense-amplitude V1 selection.
- 1 terminal raw entry uses the audited dense-amplitude V2 selection.

Every entry has positive normalized mixture weights, a natural first component,
exact source provenance, and a deterministic schedule. Every schedule was expanded
on its actual 128-step grid and passed the common rank-one, strictly one-signed
price-driver check required by DCS.

The manifest was built from clean commit
`5601baf627873aaa56eefeaa72021bddb196de3e`. It is development-informed and
therefore requires a completely fresh pilot namespace. It authorizes a new
execution configuration only; final sampling and performance claims remain closed.
