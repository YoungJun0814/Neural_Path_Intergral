# Replay cost-field correction

The original replay's scalar `work=4288` mixed generated latent samples and
geometry probes and is not a terminal-potential count. Do not use it in efficiency
comparisons. Numerical replay/path-density conclusions are unaffected.
Version2 separates 4096 regenerated latents from 194 potential evaluations
(3 model-law geometry probes of dimension64 and 2 single-path terminal evaluations).
Both artifacts remain preserved. This is diagnostic replay, not new inference.
