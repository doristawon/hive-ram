"""Startup in two phases: settings first, weights after (spec 11.1).

  tail: listen        s1: connect down (retry); listen up        s0: connect down (retry)
  s0 -HELLO{s0}-> s1 -HELLO{s0,s1}-> tail     the leader checks spec 2.3 (no weights loaded yet); ERROR ends all
  tail -GO-> s1 -> s0                         prompt precision, each stage's range; every stage loads its weights
  s0 -CAP-> s1 -CAP-> tail; the leader plans -> AGREE -> s1 -> s0
  each stage builds its Engine (scrub + graphs) -> READY + BUILT towards the leader; the leader builds its own
  the leader compares BUILT, warms the whole chain up, then serves
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any

from . import proto
from .wire import Link, LinkError


def split_addr(text: str) -> tuple[str, int]:
    host, _, port = str(text).rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"{text!r}: HOST:PORT")
    return host, int(port)


def stage_info(layers, role: str, model_dir) -> proto.StageInfo:
    """This stage's HELLO entry (after env.check_env, pins.check_tree; imports the MIA modules)."""

    import torch

    from . import env, pins
    from tensorfold.families.glm5_next.cuda.weights import Config

    cfg = Config.read(model_dir)
    total = torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else 0
    return proto.StageInfo(layers=tuple(layers), role=role, host=socket.gethostname(), pid=os.getpid(),
                           settings=env.settings_vector(model_dir), versions=env.versions(pins.EXPECT_PKG_SHA),
                           eos=list(cfg.eos), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                           mem_total=int(total))


def expect(frame, op: str, where: str):
    if frame.op == proto.ERROR:
        h = frame.hdr
        raise LinkError(f"{where}: stage {h.get('stage')} failed in {h.get('where')}: {h.get('msg')}")
    if frame.op != op:
        raise proto.ProtocolError(f"{where}: expected {op}, got {frame.op}")
    return frame


# -- the leader --------------------------------------------------------------------------------------------------
def leader_listen(addr: str, *, io_timeout: float, accept_timeout: float | None) -> Link:
    """The link from the stage before: the first connection that exchanges LINK (strays are dropped: wire.Link.
    accept_linked), within ``accept_timeout``."""

    host, port = split_addr(addr)
    link = Link.accept_linked(host, port, "up", "leader", accept_timeout=accept_timeout, io_timeout=io_timeout)
    print(f"[tfpp-glm] leader: linked with the stage before it ({link.describe()})", flush=True)
    return link


def leader_hello(link: Link | None, own: proto.StageInfo, *, timeout: float | None) -> list[proto.StageInfo]:
    if link is None:
        return [own]
    f = expect(link.recv(timeout=timeout), proto.HELLO, "startup (HELLO)")
    infos = [proto.StageInfo.from_json(d) for d in f.hdr["stages"]] + [own]
    return infos


def leader_fail(link: Link | None, where: str, exc: BaseException) -> None:
    """Tell the followers (upstream, the control direction) that startup failed, then let them exit 75."""

    if link is None:
        return
    try:
        link.send(proto.ERROR, {"stage": "leader", "where": where, "msg": str(exc)[:2000]})
    except Exception:  # noqa: BLE001  (the link may be gone already)
        pass


def leader_caps(link: Link | None, own: dict, *, timeout: float | None) -> list[dict]:
    if link is None:
        return [own]
    f = expect(link.recv(timeout=timeout), proto.CAP, "startup (CAP)")
    return list(f.hdr["caps"]) + [own]


def leader_ready(link: Link | None, own: dict, *, timeout: float | None) -> list[dict]:
    if link is None:
        return [own]
    f = expect(link.recv(timeout=timeout), proto.READY, "startup (READY)")
    return list(f.hdr["built"]) + [own]


BUILT_KEYS = ("prompt_scores_cuda", "onepass_cuda", "seg_chunks_cuda", "seg_scores_cuda", "triton_ptx", "graphs")


def built_info(layers, e) -> dict:
    """Which kernel paths this stage's build actually took (a CUDA port that failed to build falls back to Triton
    silently: G/decode.py:200-212), its peak memory, and what its GPU has free once everything is built and captured
    (``free_after``: the graph pools, kernel images and scratch the geometry does not count are in by now)."""

    import torch

    from tensorfold.families.glm5_next.cuda import latent, sparse

    graphs = mtp_graphs = 0
    g = getattr(e, "graphs", None)
    if g is not None:
        graphs, mtp_graphs = len(g.main) + len(g.sparse), len(g.mtp) + len(g.sparse_mtp)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {"layers": list(layers), "prompt_scores_cuda": bool(sparse.prompt_scores_cuda()),
            "onepass_cuda": bool(latent.onepass_cuda()), "seg_chunks_cuda": bool(latent.seg_chunks_cuda()),
            "seg_scores_cuda": bool(sparse.seg_scores_cuda()), "triton_ptx": bool(sparse.triton_matches_ptx()),
            "graphs": graphs, "mtp_graphs": mtp_graphs, "max_allocated": int(torch.cuda.max_memory_allocated()),
            "allocated": int(torch.cuda.memory_allocated()), "free_after": int(torch.cuda.mem_get_info()[0])}


def compare_built(built: list[dict]) -> list[str]:
    ref = built[-1]
    out = []
    for b in built[:-1]:
        for k in BUILT_KEYS:                       # "graphs": the main and sparse ones (the leader's MTP ones apart)
            if b.get(k) != ref.get(k):
                out.append(f"stage {b.get('layers')}: {k} {b.get(k)}, leader {ref.get(k)}")
    return out


# -- a follower --------------------------------------------------------------------------------------------------
HANDSHAKE_S = 15.0          # a listening peer answers LINK at once; silence is a relay with nobody behind it yet


def connect_down(host: str, port: int, frm: str, *, connect_timeout: float, io_timeout: float,
                 handshake_s: float = HANDSHAKE_S) -> Link:
    """Connect towards the next stage and exchange LINK, again and again until ``connect_timeout``.

    Through relay_hop.ps1 the TCP connect succeeds as soon as the relay listens, whether or not the stage behind it
    does: the relay then closes the connection (EOF) or, an older relay, leaves it silent (no LINK within
    ``handshake_s``). Either is "not there yet": close, wait a second, connect again. A peer that answers with
    another protocol (a LINK of another version, a frame that is not LINK, or bytes that are no tfpp-glm frame at all,
    such as an SSH or HTTP server's) is refused at once (ProtocolError)."""

    deadline = time.monotonic() + float(connect_timeout)
    said = False
    while True:
        down = Link.connect(host, port, "down", connect_timeout=max(0.0, deadline - time.monotonic()),
                            io_timeout=io_timeout)
        try:
            down.handshake(frm, "down", timeout=max(1.0, min(handshake_s, deadline - time.monotonic())))
            return down
        except LinkError as exc:
            down.close()
            if time.monotonic() > deadline:
                raise LinkError(f"down: {host}:{port} did not answer LINK within {connect_timeout:.0f}s ({exc})") \
                    from None
            if not said:
                print(f"[tfpp-glm] down: {host}:{port} accepted but did not answer ({exc}); the next stage is not "
                      "listening behind its relay yet, retrying", flush=True)
                said = True
            time.sleep(1.0)


def follower_connect(layers, down_addr: str, listen_addr: str | None, *, connect_timeout: float,
                     io_timeout: float) -> tuple[Link, Link | None]:
    a, b = layers
    host, port = split_addr(down_addr)
    down = connect_down(host, port, f"{a}:{b}", connect_timeout=connect_timeout, io_timeout=io_timeout)
    print(f"[tfpp-glm] [{a},{b}): linked with the next stage ({down.describe()})", flush=True)
    up = None
    if a > 0:
        if not listen_addr:
            raise ValueError("a stage past layer 0 listens for the stage before it: --listen HOST:PORT")
        lh, lp = split_addr(listen_addr)
        up = Link.accept_linked(lh, lp, "up", f"{a}:{b}", accept_timeout=connect_timeout, io_timeout=io_timeout)
        print(f"[tfpp-glm] [{a},{b}): linked with the stage before it ({up.describe()})", flush=True)
    return down, up


def forward_or_fail(f, up: Link | None) -> None:
    """A control frame goes upstream before it is handled; an ERROR from the leader ends this stage too (a LinkError:
    follower.run returns 75 and serve's os._exit skips the interpreter teardown, as for any broken chain)."""

    if up is not None:
        up.send_raw(f.raw, f.payload)
    if f.op == proto.ERROR:
        h = f.hdr
        print(f"[tfpp-glm] the leader stopped the chain: {h.get('where')}: {h.get('msg')}", flush=True)
        raise LinkError(f"the leader stopped the chain: {h.get('where')}: {h.get('msg')}")


def follower_hello(down: Link, up: Link | None, own: proto.StageInfo, *, timeout: float) -> dict:
    """HELLO towards the leader (stage 0 starts it, the others append themselves), then the leader's GO."""

    if up is None:
        stages = [own.to_json()]
    else:
        f = expect(up.recv(timeout=timeout), proto.HELLO, "startup (HELLO from the stage before)")
        stages = list(f.hdr["stages"]) + [own.to_json()]
    down.send(proto.HELLO, {"stages": stages})
    f = down.recv(timeout=timeout, keep_raw=True)
    forward_or_fail(f, up)
    expect(f, proto.GO, "startup (GO)")
    return f.hdr


def follower_cap(down: Link, up: Link | None, own: dict, *, timeout: float) -> dict:
    if up is None:
        caps = [own]
    else:
        f = expect(up.recv(timeout=timeout), proto.CAP, "startup (CAP from the stage before)")
        caps = list(f.hdr["caps"]) + [own]
    down.send(proto.CAP, {"caps": caps})
    f = down.recv(timeout=timeout, keep_raw=True)
    forward_or_fail(f, up)
    expect(f, proto.AGREE, "startup (AGREE)")
    return f.hdr


def follower_ready(down: Link, up: Link | None, own: dict, *, timeout: float) -> None:
    if up is None:
        built = [own]
    else:
        f = expect(up.recv(timeout=timeout), proto.READY, "startup (READY from the stage before)")
        built = list(f.hdr["built"]) + [own]
    down.send(proto.READY, {"built": built})


def stage_of(go: dict, layers) -> tuple[int, int]:
    """(index, count) of this follower's range in GO's chain."""

    ranges = [tuple(r) for r in go["stages"]]
    me = tuple(layers)
    if me not in ranges:
        raise proto.ProtocolError(f"the leader's chain {ranges} does not hold this stage's layers {list(me)}")
    return ranges.index(me), len(ranges)


def report_error(link: Link | None, stage: Any, where: str, exc: BaseException) -> None:
    """A follower's failure, sent towards the leader (best effort)."""

    if link is None:
        return
    try:
        link.send(proto.ERROR, {"stage": stage, "where": where, "msg": f"{type(exc).__name__}: {exc}"[:2000]})
    except Exception:  # noqa: BLE001
        pass
