# Cross-Platform Artifact Byte Stability

Frozen experiment ledgers bind generated JSON artifacts by exact SHA-256.  Git text
normalization must therefore not rewrite those artifact bytes during commit or
checkout.

Source, configuration, and documentation files remain LF-normalized.  Generated
files under `results/` and generated configuration receipts are marked `-text` in
`.gitattributes`.  This preserves their producer newline convention as part of the
immutable artifact.  JSON remains ordinary UTF-8 and is parsed normally; only Git
end-of-line conversion is disabled for byte-addressed outputs.

This policy fixes the historical failure mode in which a Windows process generated
CRLF JSON and recorded its hash, Git stored an LF-normalized blob, and a Linux CI
checkout correctly rejected the altered bytes.  Hash comparisons remain exact;
the implementation does not accept multiple hashes or normalize content at audit
time.
