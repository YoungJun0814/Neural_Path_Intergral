# G11 V8 R2j cell-tuned pilot authorization

Date: 2026-07-31  
Gate: production-size benchmark and implementation freeze  
Decision: fresh formal pilot authorized on the local CPU

The V4 benchmark executed six production-size observations, including the two
largest seven/eight-expert proposals, with 32,768 paths per observation and 16
PyTorch CPU threads. The clean benchmark passed its finite-result, seed-roster, and
resource checks.

The conservative local pilot forecast is 1,754.49 seconds for all 12,582,912 pilot
paths, below the frozen four-hour limit. Worst-case final execution remains above
the local and planning-only external limits, so only the pilot is authorized.

The V2 implementation manifest hashes the proposal manifest, mixture likelihood,
FFT simulator, DCS marginalization, controls, reference protocol, shard runners,
aggregation, and execution configuration. The pilot must run from one clean
committed source state into immutable, restart-safe shards under namespace
`v8-r2-reference-cell-tuned-pilot-v1`.

Final execution, reference completion, submission, and performance claims remain
closed until the fresh pilot allocation passes.
