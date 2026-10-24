"""tfpp-glm: GLM-5.3-Flash on the MIA package (TensorFold v0.6.0 + 107 patches) split by layer ranges over N1X laptops.

Every stage runs the package's own world=1 code on a contiguous layer range; the last stage (the leader, MIA's rank 0)
holds the head, the drafters and the HTTP front. Between stages only the residual streams (and the DFlash2 taps)
travel, bf16 as computed, so an N-stage chain equals one world=1 process bit for bit. Spec: the merged final spec of
2026-10-23 (IMPLEMENTATION_NOTES.md). Nothing here imports torch or the MIA modules at import time.
"""

PROTO = 1                 # wire protocol version (LINK frames carry it)
__version__ = "0.1.0"     # v1: one stream (PARALLEL=1)
