# G11 V8 R2M — Production-scale proposal serialization failure

Date: 2026-07-31

## Decision

The V1 production-scale training and validation namespaces are burned.  No
candidate was selected or promoted, no manifest is authorized, and no formal
pilot namespace is open.

## Failure

The frozen computation completed candidate evaluation but strict JSON
serialization rejected a positive-infinity diagnostic.  A block containing no
absolute event contribution has an undefined maximum-contribution share.  The
implementation correctly treated that case as a failing concentration value
internally, but attempted to serialize the internal `inf` while
`allow_nan=False` was active.

This is a diagnostic representation defect, not permission to relax strict
serialization.  JSON non-finite values remain forbidden.

## Recovery contract

V2 must:

1. retain positive infinity internally so the relevant candidate gate fails;
2. emit `null` for every undefined scalar diagnostic;
3. retain explicit Boolean gates so `null` can never be interpreted as a pass;
4. recursively assert that the complete result contains no non-finite float
   before writing;
5. bind the V1 execution-failure receipt by file hash; and
6. use new training and validation namespaces.

The V1 outcome object was never persisted and candidate/performance outcomes
were not inspected.  Nevertheless, the conservative namespace rule is applied
and V1 seeds will not be reused.
