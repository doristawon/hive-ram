"""One full-duplex TCP link a hop (the upstream end connects, the downstream end listens) and the ACT transfers.

Frames are proto's ``[u32 header length][JSON header][payload]``. ACT parts (the residual streams and DFlash2 taps)
go through one pinned staging buffer per link end and direction, sized once after AGREE (a prompt chunk's rows):
sending copies the GPU rows into it (the copy waits for the stream's kernels) and sends from it; receiving reads into
it and queues a non-blocking H2D copy into the destination view, whose event the next read into the buffer waits for.
Other parts (token ids, image rows) arrive as host bytes. Every frame carries the sender's sequence number on that
link and direction; a gap, a timeout or a closed peer is a ``LinkError`` and the stage stops (fail-stop).
"""

from __future__ import annotations

import errno
import select
import socket
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from . import PROTO, proto

_TORCH_DTYPE = {"bf16": torch.bfloat16, "i32": torch.int32, "f32": torch.float32, "u8": torch.uint8,
                "i64": torch.int64}
_NAME = {v: k for k, v in _TORCH_DTYPE.items()}
_NP_NAME = {np.dtype(np.int32): "i32", np.dtype(np.float32): "f32", np.dtype(np.uint8): "u8",
            np.dtype(np.int64): "i64"}
SMALL = 1 << 20             # without sendmsg (Windows tests): frames under this go out in one sendall
IOV_MAX = 64                # buffers a sendmsg call takes at most (a frame has the header and a few parts)
LINK_MAX_HEADER = 64 << 10  # a LINK header is ~80 bytes: anything this large is not a tfpp-glm peer
ACCEPT_HANDSHAKE_S = 5.0    # a connection accepted by accept_linked has this long for its LINK (connect_down waits 15)


class LinkError(RuntimeError):
    """The link broke: closed, timed out, out of sequence, or the peer reported an error."""


_align = proto.align


def _byte_view(v: memoryview) -> memoryview:
    """A flat byte view (sendmsg slices it by bytes)."""

    return v if v.format in ("B", "b", "c") and v.ndim == 1 else v.cast("B")


@dataclass
class Frame:
    hdr: dict
    raw: bytes                                   # the header as received (forward_raw sends it unchanged)
    host: dict = field(default_factory=dict)     # part name -> bytes (parts not received into a view)
    payload: list = field(default_factory=list)  # every part's bytes in order, when kept for forwarding

    @property
    def op(self) -> str:
        return self.hdr["op"]

    def ints(self, name: str) -> np.ndarray:
        dtype, shape = self._spec(name)
        if dtype != "i32":
            raise proto.ProtocolError(f"part {name} is {dtype}, not i32")
        return np.frombuffer(self.host[name], dtype=np.int32).reshape(shape)

    def tensor(self, name: str) -> torch.Tensor:
        dtype, shape = self._spec(name)
        buf = bytearray(self.host[name])
        if not buf:
            return torch.empty(shape, dtype=_TORCH_DTYPE[dtype])
        return torch.frombuffer(buf, dtype=_TORCH_DTYPE[dtype]).reshape(shape)

    def _spec(self, name: str):
        for n, d, s in self.hdr["parts"]:
            if n == name:
                return d, s
        raise proto.ProtocolError(f"{self.op} frame without part {name!r}")


class Staging:
    """A pinned host buffer and the event of the last H2D copy out of it."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = int(nbytes)
        self.t = torch.empty((max(1, self.nbytes),), dtype=torch.uint8, pin_memory=torch.cuda.is_available())
        self.np = self.t.numpy()
        self.event = None

    def wait(self) -> None:
        if self.event is not None:
            self.event.synchronize()
            self.event = None


class Link:
    def __init__(self, sock: socket.socket, name: str, *, io_timeout: float = 600.0) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # SO_SNDBUF / SO_RCVBUF are left alone: on Linux (WSL2) setting them caps both at net.core.[wr]mem_max
        # (~416 KiB by default) and turns TCP autotuning off, which would halve a prompt chunk's rate over USB4
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 6)):
            if hasattr(socket, opt):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), value)
        self.sock, self.name = sock, name
        self.io_timeout = float(io_timeout)
        self.send_seq = 0
        self.recv_seq = 0
        self.lock = threading.Lock()
        self.send_stage: Staging | None = None
        self.recv_stage: Staging | None = None
        self.sent = self.received = 0
        self.peer = ""

    def describe(self) -> str:
        """The peer and the socket buffers the kernel gave this link (autotuned: they grow under load)."""

        try:
            snd = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            rcv = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        except OSError:
            return self.peer
        return f"{self.peer}; socket buffers {snd >> 10} / {rcv >> 10} KiB, autotuned"

    # -- connecting --------------------------------------------------------------------------------------------
    @classmethod
    def listen(cls, host: str, port: int, name: str, *, accept_timeout: float | None = None,
               io_timeout: float = 600.0) -> "Link":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, int(port)))
        s.listen(1)
        s.settimeout(accept_timeout)
        print(f"[tfpp-glm] {name}: listening on {host}:{port}", flush=True)
        try:
            conn, addr = s.accept()
        except socket.timeout:
            raise LinkError(f"{name}: nobody connected to {host}:{port} within {accept_timeout:.0f}s") from None
        finally:
            s.close()
        conn.settimeout(None)
        link = cls(conn, name, io_timeout=io_timeout)
        link.peer = f"{addr[0]}:{addr[1]}"
        return link

    @classmethod
    def accept_linked(cls, host: str, port: int, name: str, frm: str, *, accept_timeout: float | None = None,
                      io_timeout: float = 600.0, handshake_s: float = ACCEPT_HANDSHAKE_S) -> "Link":
        """Listen on host:port until a tfpp-glm stage has exchanged LINK on a connection, then stop listening.

        Whatever else connects first (a port probe, a health check, a scanner, a client that stays silent or sends
        something other than a LINK frame) is closed and the listener goes on accepting: it can neither take the
        stage's one link nor hold the startup. Each accepted connection gets ``handshake_s`` for its LINK (well under
        connect_down's HANDSHAKE_S, so the real peer's own wait does not run out behind a silent one); one that sent
        its LINK and closed again (a peer that gave up and reconnects) is passed over too. A LINK of another protocol
        version comes from a real but mismatched stage: ProtocolError at once. ``accept_timeout``: the overall wait
        (None: forever)."""

        deadline = None if accept_timeout is None else time.monotonic() + float(accept_timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
            s.listen(8)
            print(f"[tfpp-glm] {name}: listening on {host}:{port}", flush=True)
            while True:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    raise LinkError(f"{name}: no tfpp-glm stage linked on {host}:{port} within {accept_timeout:.0f}s")
                s.settimeout(left)
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    raise LinkError(f"{name}: no tfpp-glm stage linked on {host}:{port} within "
                                    f"{accept_timeout:.0f}s") from None
                conn.settimeout(None)
                link = cls(conn, name, io_timeout=io_timeout)
                link.peer = f"{addr[0]}:{addr[1]}"
                t = float(handshake_s) if left is None else max(0.5, min(float(handshake_s), left))
                try:
                    why = link._try_link(frm, "up", t)
                except BaseException:
                    link.close()
                    raise
                if why is None:
                    return link
                link.close()
                print(f"[tfpp-glm] {name}: dropped a connection from {link.peer} ({why}); still listening",
                      flush=True)
        finally:
            s.close()

    def _try_link(self, frm: str, to: str, t: float) -> str | None:
        """The listening end's LINK exchange on a fresh connection, bounded by ``t`` (the first byte, then the rest of
        the frame): None when the peer is a tfpp-glm stage of this protocol, else why not (the caller drops it). A
        LINK of another protocol version raises ProtocolError."""

        saved = self.io_timeout
        self.io_timeout = float(t)
        try:
            self._send_frame(proto.make_header(proto.LINK, -1, {"proto": PROTO, "from": frm, "to": to}, []), [])
            f = self.recv(timeout=t, link_frame=True)
        except (LinkError, proto.ProtocolError, OSError, ValueError) as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            self.io_timeout = saved
        if f.op != proto.LINK:
            return f"a {f.op!r} frame instead of LINK"
        if f.hdr.get("proto") != PROTO:
            raise proto.ProtocolError(f"{self.name}: the peer speaks {f.hdr.get('proto')!r}, not tfpp-glm "
                                      f"protocol {PROTO}")
        if self._peer_gone():
            return "it closed right after its LINK"
        return None

    def _peer_gone(self) -> bool:
        """The peer has closed this connection (EOF or a reset waiting to be read)."""

        try:
            r, _, _ = select.select([self.sock], [], [], 0)
            if not r:
                return False
            return self.sock.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    @classmethod
    def connect(cls, host: str, port: int, name: str, *, connect_timeout: float = 3600.0,
                io_timeout: float = 600.0) -> "Link":
        deadline = time.monotonic() + float(connect_timeout)
        said = False
        while True:
            try:
                s = socket.create_connection((host, int(port)), timeout=10)
                s.settimeout(None)
                link = cls(s, name, io_timeout=io_timeout)
                link.peer = f"{host}:{port}"
                return link
            except OSError as exc:
                if time.monotonic() > deadline:
                    raise LinkError(f"{name}: could not connect to {host}:{port} within {connect_timeout:.0f}s "
                                    f"({exc})") from None
                if not said:
                    print(f"[tfpp-glm] {name}: waiting for {host}:{port} ({exc.strerror or exc})", flush=True)
                    said = True
                time.sleep(1.0)

    def handshake(self, frm: str, to: str, *, timeout: float = 600.0) -> dict:
        """Both ends send LINK and read the other's: the protocol version and who is on the other end."""

        self._send_frame(proto.make_header(proto.LINK, -1, {"proto": PROTO, "from": frm, "to": to}, []), [])
        f = self.recv(timeout=timeout, link_frame=True)
        if f.op != proto.LINK or f.hdr.get("proto") != PROTO:     # not a LinkError: a connect retry would not help
            raise proto.ProtocolError(f"{self.name}: the peer speaks {f.hdr.get('proto')!r}, not tfpp-glm "
                                      f"protocol {PROTO}")
        return f.hdr

    def set_staging(self, send_bytes: int = 0, recv_bytes: int = 0) -> int:
        """Pinned staging for ACT parts, allocated once (AGREE); returns the bytes allocated."""

        n = 0
        if send_bytes and self.send_stage is None:
            self.send_stage = Staging(send_bytes)
            n += self.send_stage.nbytes
        if recv_bytes and self.recv_stage is None:
            self.recv_stage = Staging(recv_bytes)
            n += self.recv_stage.nbytes
        return n

    # -- sending -----------------------------------------------------------------------------------------------
    def send(self, op: str, fields: dict | None = None, parts: Sequence[tuple[str, Any]] = ()) -> None:
        """``parts``: (name, tensor or numpy array); CUDA tensors go through the send staging (or a host copy
        without one), host ones as they are."""

        with self.lock:
            descs, views = [], []
            off = 0
            for name, t in parts:
                if isinstance(t, np.ndarray):
                    a = np.ascontiguousarray(t)
                    if a.dtype not in _NP_NAME:
                        raise proto.ProtocolError(f"part {name}: numpy dtype {a.dtype}")
                    descs.append((name, _NP_NAME[a.dtype], list(a.shape)))
                    views.append(memoryview(a.reshape(-1).view(np.uint8)))
                    continue
                if t.dtype not in _NAME:
                    raise proto.ProtocolError(f"part {name}: dtype {t.dtype}")
                descs.append((name, _NAME[t.dtype], list(t.shape)))
                flat = t.detach().contiguous().view(torch.uint8).reshape(-1) if t.numel() else \
                    torch.empty((0,), dtype=torch.uint8, device=t.device)
                n = flat.numel()
                if t.is_cuda and self.send_stage is not None and off + n <= self.send_stage.nbytes:
                    self.send_stage.t[off:off + n].copy_(flat)          # waits for the stream's kernels
                    views.append(memoryview(self.send_stage.np[off:off + n]))
                    off += _align(n)
                else:
                    views.append(memoryview(flat.cpu().numpy()))
            hdr = proto.make_header(op, self.send_seq, fields, descs)
            self._send_frame(hdr, views)
            self.send_seq += 1

    def _send_frame(self, hdr: dict, views: Sequence[memoryview]) -> None:
        raw = proto.encode_header(hdr)
        self._send_raw(raw, views)

    def send_raw(self, raw: bytes, payload: Sequence[bytes]) -> None:
        """A frame received on another link, unchanged (a follower forwarding a control op upstream)."""

        with self.lock:
            self._send_raw(raw, [memoryview(p) for p in payload])
            self.send_seq += 1

    def _send_raw(self, raw: bytes, views: Sequence[memoryview]) -> None:
        head = proto.U32.pack(len(raw)) + raw
        total = len(head) + sum(v.nbytes for v in views)
        self.sock.settimeout(self.io_timeout)
        try:
            if hasattr(self.sock, "sendmsg"):
                # scatter-gather: the header and every part straight from their buffers (the pinned staging), so a
                # decode window's ACT is not copied twice more (tobytes + join) before it leaves
                self._sendmsg_all([memoryview(head)] + [_byte_view(v) for v in views if v.nbytes])
            elif total < SMALL:
                self.sock.sendall(b"".join([head, *(v.tobytes() for v in views)]))
            else:
                self.sock.sendall(head)
                for v in views:
                    if v.nbytes:
                        self.sock.sendall(v)
        except socket.timeout:
            raise LinkError(f"{self.name}: send timed out after {self.io_timeout:.0f}s") from None
        except OSError as exc:
            raise LinkError(f"{self.name}: send failed ({exc})") from None
        finally:
            self.sock.settimeout(None)
        self.sent += total

    def _sendmsg_all(self, bufs: list) -> None:
        """sendmsg until every buffer is out (a call may send part of them)."""

        while bufs:
            n = self.sock.sendmsg(bufs[:IOV_MAX])
            if n <= 0:
                raise LinkError(f"{self.name}: send made no progress")
            while n > 0:
                if n >= bufs[0].nbytes:
                    n -= bufs[0].nbytes
                    bufs.pop(0)
                else:
                    bufs[0] = bufs[0][n:]
                    n = 0

    # -- receiving ---------------------------------------------------------------------------------------------
    def _exact(self, view: memoryview) -> None:
        got = 0
        while got < view.nbytes:
            try:
                n = self.sock.recv_into(view[got:])
            except socket.timeout:
                raise LinkError(f"{self.name}: no data for {self.sock.gettimeout():g}s mid-frame") from None
            except OSError as exc:
                raise LinkError(f"{self.name}: receive failed ({exc})") from None
            if n == 0:
                raise LinkError(f"{self.name}: the peer closed the link")
            got += n
        self.received += got

    def recv(self, *, timeout: float | None = None, into: dict | None = None, keep_raw: bool = False,
             link_frame: bool = False) -> Frame:
        """The next frame. ``timeout``: seconds to wait for its first byte (None: forever); the rest of the frame
        within io_timeout. ``into``: part name -> device view the part is copied into (ACT); ``keep_raw``: keep
        every part's bytes for ``send_raw``. ``link_frame``: the LINK exchange (no sequence number); a peer whose
        first frame cannot be a LINK (a header size out of range, parts) is not a tfpp-glm stage: ProtocolError."""

        head = bytearray(4)
        self.sock.settimeout(timeout)
        try:
            n = self.sock.recv_into(memoryview(head), 4)
        except socket.timeout:
            self.sock.settimeout(None)
            raise LinkError(f"{self.name}: nothing received within {timeout:g}s") from None
        except OSError as exc:
            self.sock.settimeout(None)
            raise LinkError(f"{self.name}: receive failed ({exc})") from None
        if n == 0:
            self.sock.settimeout(None)
            raise LinkError(f"{self.name}: the peer closed the link")
        self.sock.settimeout(self.io_timeout)
        try:
            if n < 4:
                self._exact(memoryview(head)[n:])
            self.received += n
            (size,) = proto.U32.unpack(bytes(head))
            if link_frame and not 0 < size <= LINK_MAX_HEADER:     # e.g. an SSH or TLS banner: no retry helps
                raise proto.ProtocolError(f"{self.name}: a {size}-byte first frame header: the peer does not speak "
                                          f"tfpp-glm protocol {PROTO}")
            if not 0 < size <= proto.MAX_HEADER:
                raise LinkError(f"{self.name}: a {size}-byte frame header")
            raw = bytearray(size)
            self._exact(memoryview(raw))
            raw = bytes(raw)
            hdr = proto.decode_header(raw)
            frame = Frame(hdr, raw)
            if link_frame and hdr["parts"]:
                raise proto.ProtocolError(f"{self.name}: a {hdr['op']} frame with parts during the LINK exchange")
            if not link_frame:
                if hdr["seq"] != self.recv_seq:
                    raise LinkError(f"{self.name}: frame {hdr['op']} seq {hdr['seq']}, expected {self.recv_seq}")
                self.recv_seq += 1
            off = 0
            copied = False
            for name, dtype, shape in hdr["parts"]:
                nbytes = proto.part_nbytes(dtype, shape)
                dst = None if into is None else into.get(name)
                if dst is not None:
                    if keep_raw:
                        raise proto.ProtocolError("a forwarded frame cannot land in device views")
                    if _NAME.get(dst.dtype) != dtype or list(dst.shape) != list(shape) or not dst.is_contiguous():
                        raise LinkError(f"{self.name}: part {name} {dtype}{shape} does not fit its destination "
                                        f"{dst.dtype}{list(dst.shape)}")
                    st = self.recv_stage
                    if st is None or off + nbytes > st.nbytes:
                        tmp = bytearray(nbytes)
                        self._exact(memoryview(tmp))
                        if nbytes:
                            dst.copy_(torch.frombuffer(tmp, dtype=_TORCH_DTYPE[dtype]).reshape(shape))
                        continue
                    if not copied:
                        st.wait()                      # the last frame's H2D copies are done with the buffer
                    self._exact(memoryview(st.np[off:off + nbytes]))
                    if nbytes:
                        src = st.t[off:off + nbytes].view(_TORCH_DTYPE[dtype]).reshape(shape)
                        dst.copy_(src, non_blocking=True)
                        copied = True
                    off += _align(nbytes)
                else:
                    buf = bytearray(nbytes)
                    self._exact(memoryview(buf))
                    frame.host[name] = bytes(buf)
                    if keep_raw:
                        frame.payload.append(frame.host[name])
            if copied:
                ev = torch.cuda.Event()
                ev.record()
                self.recv_stage.event = ev
            return frame
        finally:
            self.sock.settimeout(None)

    def readable(self, timeout: float = 0.0) -> bool:
        """Data (or a closed peer) to read within ``timeout`` seconds."""

        try:
            r, _, _ = select.select([self.sock], [], [], timeout)
        except InterruptedError:
            return False
        return bool(r)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def wait_any(links: Sequence[Link], timeout: float | None = None) -> Link | None:
    """The first of ``links`` with data (or a closed peer) to read; None on timeout."""

    socks = {l.sock: l for l in links if l is not None}
    while True:
        try:
            r, _, _ = select.select(list(socks), [], [], timeout)
        except InterruptedError:
            continue
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            raise
        return socks[r[0]] if r else None


staging_bytes = proto.link_staging


class ActIO:
    """A stage's ACT traffic: the residual streams (and the carried taps) in from upstream, out to downstream."""

    def __init__(self, stage, up: Link | None, down: Link | None, *, warm=None, timeout: float = 600.0,
                 tracer=None) -> None:
        self.stage, self.up, self.down = stage, up, down
        self.warm = warm
        self.timeout = float(timeout)
        self.tracer = tracer
        self.outstanding = 0          # the leader: ACT frames asked for (PREFILL chunks, STEPs) and not yet read
        self.recv_ms = 0.0

    def _views(self, b, R: int, slots) -> dict:
        into = {"x": b.x[:R]}
        for s in slots:
            into[f"tap:{s}"] = b.taps[s][:R]
        return into

    def recv_act(self, b, R: int, st, kind: str) -> None:
        S = self.stage
        if self.up is None:
            raise proto.ProtocolError("recv_act on stage 0")
        into = self._views(b, R, S.carry_in)
        t = time.perf_counter()
        with (self.warm if self.warm is not None else nullcontext()):
            f = self.up.recv(timeout=self.timeout, into=into)
        self.recv_ms += (time.perf_counter() - t) * 1e3
        if f.op == proto.ERROR:
            raise LinkError(f"stage {f.hdr.get('stage')} failed in {f.hdr.get('where')}: {f.hdr.get('msg')}")
        h = f.hdr
        if f.op != proto.ACT or h.get("kind") != kind or h.get("R") != R or h.get("pos") != st.pos:
            raise LinkError(f"{self.up.name}: expected ACT {kind} R={R} pos={st.pos}, got {f.op} {h.get('kind')} "
                            f"R={h.get('R')} pos={h.get('pos')} (the chain lost step)")
        names = {p[0] for p in h["parts"]}
        if names != set(into):
            raise LinkError(f"{self.up.name}: ACT parts {sorted(names)}, expected {sorted(into)}")
        if self.outstanding > 0:
            self.outstanding -= 1

    def send_act(self, b, R: int, st, kind: str) -> None:
        S = self.stage
        parts = [("x", b.x[:R])] + [(f"tap:{s}", b.taps[s][:R]) for s in S.carry_out]
        self.down.send(proto.ACT, {"kind": kind, "pos": int(st.pos), "R": int(R)}, parts)
        if self.tracer is not None:
            self.tracer.act(kind, int(st.pos), int(R), b)
