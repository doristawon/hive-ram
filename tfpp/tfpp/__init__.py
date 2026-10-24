# SPDX-License-Identifier: Apache-2.0
"""tfpp: pipeline parallelism for TensorFold 0.6.1's CUDA Flash Next (Apache-2.0), across machines over TCP.

Each rank holds a contiguous range of decoder layers. Rank 0 embeds and runs the first range; the last rank runs
the head, samples, verifies MTP drafts and drafts the next ones. Between ranks only the residual streams travel
(bf16, streams x hidden per row), once per round in each direction.
"""

__version__ = "0.2.0"     # 0.2: link handshake, HELLO chain check, checked frames, error frames (wire tfpp-flashnext/2)
