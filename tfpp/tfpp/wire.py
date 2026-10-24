# SPDX-License-Identifier: Apache-2.0
"""Framing between ranks: [u32 header length][header JSON][payload]; the payload is one bf16 tensor or nothing.

Linking. The listening end (the rank after) sends ``ready`` (protocol, nonce) as soon as it accepts; the connecting
end answers ``link`` and the listening end confirms with ``linked`` (or ``refused``). ``connect`` retries until that
exchange completes, so the ranks may start in any order, and a relay that accepts with nobody behind it yet (the WSL
NAT relay on the Windows side) counts as "not there yet" instead of a link that never answers. A listening end that
gets something else (a port scanner, linktest.py) drops it and keeps listening.

Trust model: like llama.cpp's rpc-server, a rank link is meant for a private point-to-point link (the USB4 cable).
Nothing in a frame can run code (the headers are JSON) and every size and token id is checked before anything is
allocated or launched, but by default anything that speaks the protocol may link. Keep the port off LANs and WiFi:
bind the link address (``TFPP_BIND`` / ``--bind``; the default 0.0.0.0 is what a WSL2-NAT node behind its relay
needs) and allow the port only from the peer in the firewall. Optional: set the same ``TFPP_SECRET`` on every node
and both ends of every link prove they know it (HMAC-SHA256 over both nonces); a peer that does not is refused.

Timeouts. Sockets use TCP keepalive. A frame's first byte is awaited for as long as the caller says (forever while a
follower is idle between requests), the rest of a frame and every send within ``io_timeout`` (``TFPP_IO_TIMEOUT``,
default 600 s; 0 turns it off). A timeout, a closed peer or a malformed frame raises ``LinkError``
(``ProtocolError`` for a malformed one), a ``ConnectionError``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import select
import socket
import struct
import time
from math import prod

try:                                   # the framing and the link handshake are testable without torch
    import torch
except ImportError:                    # pragma: no cover - CPU test machines
    torch = None

from .proto import PROTO, LinkError, ProtocolError, RemoteError  # noqa: F401 (re-exported)

_U32 = struct.Struct("<I")
MAX_HEADER = 1 << 20                   # a prompt chunk's header (2048 ids twice) is about 30 KB
MAX_PAYLOAD = 256 << 20                # until the rank sets its own cap (prompt chunk rows x streams x hidden x 2)
SMALL = 1 << 20                        # frames under this go out in one sendall
HANDSHAKE_S = 15.0                     # a listening rank answers at once; silence is a relay with nobody behind it
CONNECT_TIMEOUT = 3600.0
DEFAULT_BIND = os.environ.get("TFPP_BIND", "0.0.0.0")
_KEEP = object()                       # recv(timeout=...): the link's io_timeout


def _env_timeout() -> float | None:
    try:
        v = float(os.environ.get("TFPP_IO_TIMEOUT", "600"))
    except ValueError:
        v = 600.0
    return v if v > 0 else None


IO_TIMEOUT = _env_timeout()


def _secret() -> bytes:
    return os.environ.get("TFPP_SECRET", "").encode()


def _mac(secret: bytes, what: str, n1: str, n2: str) -> str:
    return hmac.new(secret, f"{PROTO}|{what}|{n1}|{n2}".encode(), hashlib.sha256).hexdigest()


def _nonce(hdr: dict) -> str:
    n = hdr.get("nonce")
    if not isinstance(n, str) or not 16 <= len(n) <= 128:
        raise ProtocolError("a link handshake without a nonce")
    return n


def check_payload(header: dict, max_payload: int) -> list[int]:
    """The tensor a header announces: bf16 of ``shape`` (non-negative ints) in exactly ``nbytes`` bytes, at most
    ``max_payload``. Checked before the pinned buffer is allocated."""

    n, shape = header.get("nbytes"), header.get("shape")
    if type(n) is not int or n <= 0:     # noqa: E721
        raise ProtocolError(f"a payload of {n!r} bytes")
    if n > max_payload:
        raise ProtocolError(f"a {n}-byte payload; this link takes at most {max_payload}")
    if not isinstance(shape, list) or not shape or not all(type(d) is int and d >= 0 for d in shape):  # noqa: E721
        raise ProtocolError(f"a payload of shape {shape!r}")
    if 2 * prod(shape) != n:
        raise ProtocolError(f"a bf16 payload of shape {shape} in {n} bytes")
    return shape


def _keepalive(sock: socket.socket) -> None:
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 6)):
            if hasattr(socket, opt):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), value)
    except OSError:
        pass


class Link:
    def __init__(self, sock: socket.socket, *, name: str = "link", io_timeout: float | None = IO_TIMEOUT,
                 max_payload: int = MAX_PAYLOAD) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # SO_SNDBUF / SO_RCVBUF are left to the kernel: on Linux (WSL2) setting them caps both at
        # 2 x net.core.[wr]mem_max (about 416 KiB by default) and turns TCP autotuning off
        _keepalive(sock)
        self.sock = sock
        self.name = name
        self.io_timeout = io_timeout or None
        self.max_payload = int(max_payload)  # the largest tensor this end accepts (0: none)
        self.host = {}                       # pinned staging buffers by byte size
        self.sent = self.received = 0
        self.peer = ""
        sock.settimeout(self.io_timeout)

    def describe(self) -> str:
        """The peer and the socket buffers the kernel gave this link (autotuned: they grow under load)."""

        try:
            snd = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            rcv = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        except OSError:
            return self.peer
        return f"{self.peer}; socket buffers {snd >> 10} / {rcv >> 10} KiB"

    # -- linking --------------------------------------------------------------------------------------------------
    @classmethod
    def listen(cls, port: int, host: str | None = None, *, name: str = "up", accept_timeout: float | None = None,
               secret: bytes | None = None, io_timeout: float | None = IO_TIMEOUT,
               max_payload: int = MAX_PAYLOAD) -> "Link":
        """Wait for the rank before this one on ``host:port`` (default ``TFPP_BIND`` or 0.0.0.0)."""

        host = DEFAULT_BIND if host is None else host
        secret = _secret() if secret is None else secret
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, int(port)))
        s.listen(4)
        print(f"[tfpp] {name}: listening on {host}:{port}{' (TFPP_SECRET set)' if secret else ''}", flush=True)
        deadline = None if accept_timeout is None else time.monotonic() + float(accept_timeout)
        try:
            while True:
                s.settimeout(None if deadline is None else max(0.001, deadline - time.monotonic()))
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    raise LinkError(f"{name}: nobody linked on {host}:{port} within {accept_timeout:g}s") from None
                link = cls(conn, name=name, io_timeout=io_timeout, max_payload=max_payload)
                link.peer = f"{addr[0]}:{addr[1]}"
                try:
                    link._answer(secret)
                    return link
                except (LinkError, OSError, ValueError) as exc:
                    print(f"[tfpp] {name}: dropped a connection from {link.peer} ({exc}); still listening",
                          flush=True)
                    link.close()
        finally:
            s.close()

    @classmethod
    def connect(cls, host: str, port: int, *, name: str = "down", connect_timeout: float = CONNECT_TIMEOUT,
                secret: bytes | None = None, io_timeout: float | None = IO_TIMEOUT,
                max_payload: int = MAX_PAYLOAD, handshake_s: float = HANDSHAKE_S) -> "Link":
        """Link to the rank after this one, again and again until it answers or ``connect_timeout`` passes."""

        secret = _secret() if secret is None else secret
        deadline = time.monotonic() + float(connect_timeout)
        said = False
        while True:
            try:
                s = socket.create_connection((host, int(port)), timeout=10)
            except OSError as exc:
                reason = exc.strerror or str(exc)
            else:
                link = cls(s, name=name, io_timeout=io_timeout, max_payload=max_payload)
                link.peer = f"{host}:{port}"
                try:
                    link._greet(secret, handshake_s)
                    return link
                except ProtocolError:
                    link.close()
                    raise
                except (LinkError, OSError) as exc:
                    link.close()
                    reason = str(exc)
            if time.monotonic() >= deadline:
                raise LinkError(f"{name}: could not link to {host}:{port} within {connect_timeout:g}s ({reason})")
            if not said:
                print(f"[tfpp] {name}: waiting for {host}:{port} ({reason})", flush=True)
                said = True
            time.sleep(1.0)

    def _answer(self, secret: bytes) -> None:
        n1 = secrets.token_hex(16)
        self.sock.settimeout(HANDSHAKE_S)
        try:
            self._send_header({"op": "ready", "proto": PROTO, "nonce": n1})
            hdr = self._recv_header(HANDSHAKE_S)
            if hdr.get("op") != "link" or hdr.get("proto") != PROTO:
                raise ProtocolError(f"not a {PROTO} rank ({hdr.get('op')!r}, {hdr.get('proto')!r})")
            n2 = _nonce(hdr)
            if secret and not hmac.compare_digest(str(hdr.get("mac", "")).encode(), _mac(secret, "link", n1, n2)
                                                  .encode()):
                self._send_header({"op": "refused", "msg": "TFPP_SECRET differs, or is unset on the connecting "
                                                           "node"})
                raise ProtocolError("the peer did not prove TFPP_SECRET")
            self._send_header({"op": "linked", "mac": _mac(secret, "linked", n1, n2) if secret else ""})
        finally:
            self.sock.settimeout(self.io_timeout)

    def _greet(self, secret: bytes, handshake_s: float) -> None:
        self.sock.settimeout(handshake_s)
        try:
            hdr = self._recv_header(handshake_s)
            if hdr.get("op") != "ready" or hdr.get("proto") != PROTO:
                raise ProtocolError(f"{self.peer} is not a {PROTO} rank ({hdr.get('op')!r}, {hdr.get('proto')!r}); "
                                    "re-sync every node")
            n1 = _nonce(hdr)
            n2 = secrets.token_hex(16)
            self._send_header({"op": "link", "proto": PROTO, "nonce": n2,
                               "mac": _mac(secret, "link", n1, n2) if secret else ""})
            hdr = self._recv_header(handshake_s)
            if hdr.get("op") == "refused":
                raise ProtocolError(f"{self.peer} refused the link: {hdr.get('msg')}")
            if hdr.get("op") != "linked":
                raise ProtocolError(f"{self.peer} answered {hdr.get('op')!r} to link")
            if secret and not hmac.compare_digest(str(hdr.get("mac", "")).encode(),
                                                  _mac(secret, "linked", n1, n2).encode()):
                raise ProtocolError(f"{self.peer} did not prove TFPP_SECRET (it differs, or is unset there)")
        finally:
            self.sock.settimeout(self.io_timeout)

    # -- sending --------------------------------------------------------------------------------------------------
    def _pinned(self, n: int):
        t = self.host.get(n)
        if t is None:
            t = self.host[n] = torch.empty((n,), dtype=torch.uint8, pin_memory=True)
        return t

    def _sendall(self, data) -> None:
        try:
            self.sock.sendall(data)
        except socket.timeout:
            raise LinkError(f"{self.name}: send timed out after {self.io_timeout:g}s") from None
        except OSError as exc:
            raise LinkError(f"{self.name}: send failed ({exc})") from None

    def _send_header(self, header: dict) -> None:
        raw = json.dumps(header, separators=(",", ":")).encode()
        if len(raw) > MAX_HEADER:
            raise ValueError(f"a {len(raw)}-byte frame header (at most {MAX_HEADER})")
        self._sendall(_U32.pack(len(raw)) + raw)
        self.sent += 4 + len(raw)

    def send(self, header: dict, tensor=None) -> None:
        if tensor is not None:
            flat = tensor.contiguous().view(torch.uint8).reshape(-1)
            host = self._pinned(flat.numel())
            host.copy_(flat)                 # waits for the producing kernels on this stream
            header = {**header, "shape": list(tensor.shape), "nbytes": flat.numel()}
        raw = json.dumps(header, separators=(",", ":")).encode()
        if len(raw) > MAX_HEADER:
            raise ValueError(f"a {len(raw)}-byte frame header (at most {MAX_HEADER})")
        parts = [_U32.pack(len(raw)), raw]
        if tensor is not None:
            parts.append(memoryview(host.numpy()))
        total = sum(len(p) for p in parts)
        if total < SMALL:                    # one segment train for a decode round's few KB
            self._sendall(b"".join(bytes(p) for p in parts))
        else:
            for p in parts:
                self._sendall(p)
        self.sent += total

    def fail(self, where: str, msg: str, *, drain_s: float = 3.0) -> None:
        """Best effort: tell the peer why this rank stops (an ``error`` frame), then (``drain_s`` > 0) close the
        sending side and read until the peer closes, so unread data does not turn the close into a reset that
        drops the frame."""

        try:
            self.sock.settimeout(5.0)
            self._send_header({"op": "error", "where": str(where), "msg": str(msg)[:4000]})
            if drain_s <= 0:
                return
            self.sock.shutdown(socket.SHUT_WR)
            end = time.monotonic() + drain_s
            buf = bytearray(1 << 16)
            while True:
                left = end - time.monotonic()
                if left <= 0:
                    break
                self.sock.settimeout(left)
                if self.sock.recv_into(buf) == 0:
                    break
        except (OSError, ValueError, LinkError):
            pass

    # -- receiving ------------------------------------------------------------------------------------------------
    def _exact(self, view: memoryview) -> None:
        got = 0
        while got < len(view):
            try:
                n = self.sock.recv_into(view[got:])
            except socket.timeout:
                raise LinkError(f"{self.name}: no data for {self.sock.gettimeout():g}s in the middle of a frame") \
                    from None
            except OSError as exc:
                raise LinkError(f"{self.name}: receive failed ({exc})") from None
            if n == 0:
                raise LinkError(f"{self.name}: peer closed the link")
            got += n
        self.received += got

    def _recv_header(self, timeout=_KEEP) -> dict:
        """The next frame's header. ``timeout``: seconds to wait for its first byte (None: forever; default: the
        link's io_timeout); the rest of the frame comes within the socket's timeout (io_timeout, or the
        handshake's while linking)."""

        rest = self.sock.gettimeout()
        wait = self.io_timeout if timeout is _KEEP else timeout
        head = bytearray(4)
        switch = wait != rest
        if switch:
            self.sock.settimeout(wait)
        try:
            n = self.sock.recv_into(head, 4)
        except socket.timeout:
            raise LinkError(f"{self.name}: nothing received within {wait:g}s") from None
        except OSError as exc:
            raise LinkError(f"{self.name}: receive failed ({exc})") from None
        finally:
            if switch:
                self.sock.settimeout(rest)
        if n == 0:
            raise LinkError(f"{self.name}: peer closed the link")
        self.received += n
        if n < 4:
            self._exact(memoryview(head)[n:])
        size = _U32.unpack(head)[0]
        if not 0 < size <= MAX_HEADER:
            raise ProtocolError(f"{self.name}: a {size}-byte frame header (at most {MAX_HEADER}): not a tfpp peer")
        raw = bytearray(size)
        self._exact(memoryview(raw))
        try:
            header = json.loads(raw)
        except ValueError:
            raise ProtocolError(f"{self.name}: a frame header that is not JSON: not a tfpp peer") from None
        if not isinstance(header, dict):
            raise ProtocolError(f"{self.name}: a frame header that is not a JSON object")
        return header

    def recv(self, device: str = "cuda", *, timeout=_KEEP):
        """The next frame: (header, bf16 tensor on ``device`` or None). ``timeout`` as in ``_recv_header``."""

        header = self._recv_header(timeout)
        if not header.get("nbytes"):
            return header, None
        shape = check_payload(header, self.max_payload)
        n = header["nbytes"]
        host = self._pinned(n)
        self._exact(memoryview(host.numpy()))
        t = host.to(device, non_blocking=False).view(torch.bfloat16).reshape(shape)
        return header, t

    def readable(self, timeout: float = 0.0) -> bool:
        """Data (or a closed peer) to read within ``timeout`` seconds."""

        try:
            r, _, _ = select.select([self.sock], [], [], timeout)
        except (OSError, ValueError):
            return True                      # a broken socket: the read that follows reports it
        return bool(r)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
