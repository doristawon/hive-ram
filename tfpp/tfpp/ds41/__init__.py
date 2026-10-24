"""tfpp-ds41: DeepSeek-V4.1-Flash on the bertholomus TensorFold fork (deepseek-v41-tp2 @ af26d42b1, world=1) split by
contiguous layer ranges over N1X laptops.

Every stage runs the fork's own world=1 code (DsEngine's ``--parallel`` path: MultiDecoder, RoundRunner, the per-stretch
round graphs) on its layer range; the last stage (the leader) holds the head, DSpark, the Scheduler and the HTTP front.
Between stages travel a stretch's state (the residual streams h, the pre-mix, and what the layers after the cut read of
the layers before it: the indexer top-k, the candidate pool, the compressed kv-source rows they attend), so an N-stage
chain equals the unmodified fork's one-process world=1 run bit for bit. Spec: the merged final spec of 2026-10-23
(IMPLEMENTATION_NOTES.md). Nothing here imports torch or the fork's modules at import time.
"""

PROTO = "tfpp-ds41/1"     # LINK frames carry it (tfpp-glm's links say 1: the two never connect)
__version__ = "0.1.0"     # v1: KEEP=0, no TOUCH frames, no vision, no spill, chunk by chunk prefill
