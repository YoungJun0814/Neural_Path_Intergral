# G11 V8 R2b reference resource decision

Date: 2026-07-31  
Benchmark class: clean-source development resource measurement  
Formal reference evidence: no

Benchmark result SHA-256:
`844ff2bcdcb442e8f554aad8ac015dfb4bdcbb9bd3b77621f3975b7722015787`  
Implementation manifest SHA-256:
`dc99812b1f8f85ea15757841a666db7d017296b29ac40a188d394570af1db5cf`  
Pilot authorization SHA-256:
`db733a636d4f21180bb8b33c691101c9add1a28ac06556b5817fa629ec9c9f9d`  
Resource audit SHA-256:
`c940615d42262a8eb63dbc18b73cbf8f023518ef4f21547b5446dba4840e820d`

## Decision

The full 384-shard pilot roster is authorized on the laptop from a clean worktree.
Final reference execution is not authorized. It requires the post-pilot allocation
manifest and a new resource review.

## Measured design

The benchmark uses the same 32,768 paths and 128 time steps as one production pilot
shard, rather than extrapolating memory from a tiny smoke batch. It covers:

- moderate terminal, \(H=0.12\), nominal \(10^{-2}\);
- rare terminal, \(H=0.05\), nominal \(10^{-5}\);
- rare discrete barrier, \(H=0.05\), nominal \(10^{-5}\); and
- both independent DCS and raw reference paths.

The source is clean at commit
`d778864952d4f33ca53a1b144e6709e08127ee50`. The benchmark namespace is burned
and cannot be reused by formal pilots.

## Conservative forecasts

| Workload | Predicted wall | Predicted peak RAM | Launch |
|---|---:|---:|---|
| laptop, complete pilots | 1,551.56 s (0.43 h) | 4.89 GB | yes |
| laptop, cap-level final | 49,649.90 s (13.79 h) | 4.89 GB | no |
| 32-vCPU planning model, pilots | 1,108.26 s (0.31 h) | 9.78 GB | planning only |
| 32-vCPU planning model, cap-level final | 35,464.21 s (9.85 h) | 9.78 GB | no |

The wall forecasts apply a factor-two safety margin and use the slowest observed
throughput. The local RAM gate compares twice the largest measured process peak
against 60% of physical RAM.

## Threading correction

One observed process already uses 16 PyTorch CPU threads. The external planning
model therefore uses two processes for a 32-vCPU node, not 32 processes. The latter
would multiply a 16-thread throughput measurement 32 times and severely overstate
parallel speed while also oversubscribing the node.

No external launch is authorized from a laptop measurement. External hardware must
run the same production-size benchmark with its own environment and thread contract.

## Formal pilot conditions

- verify every path/hash in the frozen implementation manifest;
- clean Git worktree at process start;
- CPU/float64 and 16 PyTorch threads;
- exact 24 cells, two methods, eight replicates, and 32,768 paths per shard;
- immutable output under ignored restart storage; and
- no final outcome or final allocation may be inspected during pilot execution.

Passing pilots will only authorize allocation freeze. It will not complete the
reference, authorize a V8 performance claim, or authorize P8.
