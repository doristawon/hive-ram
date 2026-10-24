"""Startup in two phases: settings first, weights after (spec 9.2); G/startup.py's shape on tfpp-ds41's links.

  leader: listen               follower: connect down (retry until LINK answers); listen up (A > 0)
  s0 -HELLO{s0}-> s1 -HELLO{s0,s1}-> leader   settings, versions, Engram ownership; no weights loaded yet
  leader -GO-> ... -> s0                      ranges, context, parallel, drafts, chunk, replay, experts, cut_post,
                                              plan digest (each stage recomputes its plan); every stage loads weights
  s0 -CAP-> ... -> leader                     budget (one mem_get_info after the weights), need, staging
  leader -AGREE-> ... -> s0                   cap, slots, staging; every stage builds (pool, warm-up, graphs)
  s0 -READY-> ... -> leader                   captured (R, b), stretches, switches, memory after the build
An ERROR in either direction stops every stage (exit 75).
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import Any

from . import proto
from .wire import DsLink, LinkError


def log(msg: str) -> None:
    print(f"[tfpp-ds41] {msg}", flush=True)


def split_addr(text: str) -> tuple[str, int]:
    host, _, port = str(text).rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"{text!r}: HOST:PORT")
    return host, int(port)


def stage_info(layers, role: str, model_dir, *, experts: int, carry: str, extra: dict) -> proto.DsStageInfo:
    """This stage's HELLO entry (after env.check_env and pins.check_tree; imports the fork's modules)."""

    import torch

    from . import env, pins

    total = torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else 0
    return proto.DsStageInfo(layers=tuple(layers), role=role, host=socket.gethostname(), pid=os.getpid(),
                             settings=env.settings_vector(model_dir, experts_k=experts, carry=carry),
                             versions=env.versions(pins.EXPECT_PKG_SHA), extra=extra,
                             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                             mem_total=int(total))


def stage_extra(model_dir, engram_dir: str | None, mv) -> dict:
    """This stage's own HELLO items (spec 13.2): its Engram tables (the layers they hold, their layout's sha, sizes,
    real or test) and its token map's sha; checked against the config here and across owners by the leader."""

    extra: dict = {"engram": None, "token_map_sha": None}
    if not engram_dir:
        return extra
    from tensorfold.families.deepseek_v41.config import Cfg

    from .load import token_map
    from .part import engram_files
    from .testmodel import AIO_PROBE_ROW, engram_layer_of, table_identity

    files = engram_files(engram_dir)
    if not files:
        raise RuntimeError(f"--engram {engram_dir}: no *.safetensors there")
    ident = table_identity(files)
    for name, rows in ident["rows"].items():
        layer = engram_layer_of(name)
        if layer not in mv.engram_layers:
            raise RuntimeError(f"{engram_dir}: a table of layer {layer}, the config's Engram layers are "
                               f"{list(mv.engram_layers)}")
        want = mv.engram_rows[list(mv.engram_layers).index(layer)]
        if int(rows) != int(want):
            raise RuntimeError(f"{engram_dir}: {name} has {rows} rows, the config says {want} (engram_num_embeddings)")
        if int(rows) <= AIO_PROBE_ROW:
            log(f"{name}: {rows} rows: the AIO self-test reads row {AIO_PROBE_ROW} (F/model.py:557); decode reads stay "
                "on the reader pool")
    extra["engram"] = ident
    _, extra["token_map_sha"] = token_map(model_dir, Cfg.read(model_dir))
    return extra


def f1_chunk_now(carry: str, mv) -> bool:
    """Whether prompt chunks carry the pending post (F1) under this process's kernel switches."""

    if carry != "carry":
        return False
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    return bool(K.on("hc_pf2") and K.hc_pf2_ok(mv.dim))


def expect(frame, op: str, where: str):
    if frame.op == proto.ERROR:
        h = frame.hdr
        raise LinkError(f"{where}: stage {h.get('stage')} failed in {h.get('where')}: {h.get('msg')}")
    if frame.op != op:
        raise proto.ProtocolError(f"{where}: expected {op}, got {frame.op}")
    return frame


# -- the leader --------------------------------------------------------------------------------------------------
def leader_listen(addr: str, *, io_timeout: float, accept_timeout: float | None) -> DsLink:
    host, port = split_addr(addr)
    link = DsLink.listen(host, port, "up", accept_timeout=accept_timeout, io_timeout=io_timeout)
    link.handshake("leader", "up", timeout=io_timeout)
    log(f"leader: linked with the stage before it ({link.describe()})")
    return link


def leader_hello(link: DsLink | None, own: proto.DsStageInfo, *, timeout: float | None) -> list[proto.DsStageInfo]:
    if link is None:
        return [own]
    f = expect(link.recv(timeout=timeout), proto.HELLO, "startup (HELLO)")
    return [proto.DsStageInfo.from_json(d) for d in f.hdr["stages"]] + [own]


def leader_fail(link: DsLink | None, where: str, exc: BaseException) -> None:
    """Tell the followers (upstream) that startup failed; they exit 75."""

    if link is None:
        return
    try:
        link.send(proto.ERROR, {"stage": "leader", "where": where, "msg": str(exc)[:2000]})
    except Exception:  # noqa: BLE001  (the link may be gone already)
        pass


def leader_collect(link: DsLink | None, op: str, key: str, own: dict, *, timeout: float | None) -> list[dict]:
    if link is None:
        return [own]
    f = expect(link.recv(timeout=timeout), op, f"startup ({op})")
    return list(f.hdr[key]) + [own]


# -- a follower --------------------------------------------------------------------------------------------------
HANDSHAKE_S = 15.0          # a listening peer answers LINK at once; silence is a relay with nobody behind it yet


def connect_down(host: str, port: int, frm: str, *, connect_timeout: float, io_timeout: float,
                 handshake_s: float = HANDSHAKE_S) -> DsLink:
    """Connect towards the next stage and exchange LINK, again and again until ``connect_timeout`` (through
    relay_hop.ps1 a TCP connect succeeds as soon as the relay listens: EOF or silence means "not there yet"); a peer
    speaking another protocol is refused at once (G/startup.py:136-162)."""

    deadline = time.monotonic() + float(connect_timeout)
    said = False
    while True:
        down = DsLink.connect(host, port, "down", connect_timeout=max(0.0, deadline - time.monotonic()),
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
                log(f"down: {host}:{port} accepted but did not answer ({exc}); retrying until the next stage listens")
                said = True
            time.sleep(1.0)


def follower_connect(layers, down_addr: str, listen_addr: str | None, *, connect_timeout: float,
                     io_timeout: float) -> tuple[DsLink, DsLink | None]:
    a, b = layers
    host, port = split_addr(down_addr)
    down = connect_down(host, port, f"{a}:{b}", connect_timeout=connect_timeout, io_timeout=io_timeout)
    log(f"[{a},{b}): linked with the next stage ({down.describe()})")
    up = None
    if a > 0:
        if not listen_addr:
            raise ValueError("a stage past layer 0 listens for the stage before it: --listen HOST:PORT")
        lh, lp = split_addr(listen_addr)
        up = DsLink.listen(lh, lp, "up", accept_timeout=connect_timeout, io_timeout=io_timeout)
        up.handshake(f"{a}:{b}", "up", timeout=connect_timeout)
        log(f"[{a},{b}): linked with the stage before it ({up.describe()})")
    return down, up


def forward_or_fail(f, up: DsLink | None) -> None:
    """A control frame goes upstream before it is handled; an ERROR from the leader ends this stage too."""

    if up is not None:
        up.send_raw(f.raw, f.payload)
    if f.op == proto.ERROR:
        h = f.hdr
        log(f"the leader stopped the chain: {h.get('where')}: {h.get('msg')}")
        sys.exit(proto.FAIL_EXIT)


def follower_exchange(down: DsLink, up: DsLink | None, op: str, key: str, own: dict, reply: str, *,
                      timeout: float) -> dict:
    """Send ``op`` towards the leader (stage 0 starts it, the others append themselves), then the leader's
    ``reply`` (forwarded upstream first)."""

    if up is None:
        items = [own]
    else:
        f = expect(up.recv(timeout=timeout), op, f"startup ({op} from the stage before)")
        items = list(f.hdr[key]) + [own]
    down.send(op, {key: items})
    f = down.recv(timeout=timeout, keep_raw=True)
    forward_or_fail(f, up)
    expect(f, reply, f"startup ({reply})")
    return f.hdr


def follower_ready(down: DsLink, up: DsLink | None, own: dict, *, timeout: float) -> None:
    if up is None:
        built = [own]
    else:
        f = expect(up.recv(timeout=timeout), proto.READY, "startup (READY from the stage before)")
        built = list(f.hdr["built"]) + [own]
    down.send(proto.READY, {"built": built})


def stage_of(go: dict, layers) -> tuple[int, int]:
    ranges = [tuple(r) for r in go["stages"]]
    me = tuple(layers)
    if me not in ranges:
        raise proto.ProtocolError(f"the leader's chain {ranges} does not hold this stage's layers {list(me)}")
    return ranges.index(me), len(ranges)


def report_error(link: DsLink | None, stage: Any, where: str, exc: BaseException) -> None:
    """A follower's failure, sent towards the leader (best effort)."""

    if link is None:
        return
    try:
        link.send(proto.ERROR, {"stage": stage, "where": where, "msg": f"{type(exc).__name__}: {exc}"[:2000]})
    except Exception:  # noqa: BLE001
        pass
