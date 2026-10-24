"""USB4 link test through the real path (client node's WSL -> relay on the server node's Windows side -> server
node's WSL). Standard library only.
  server:  python3 linktest.py server [port] [host]   (server node, inside WSL; serves one client, then exits)
  client:  python3 linktest.py client HOST [port]     (client node, inside WSL)
The server binds host, default 0.0.0.0 (the Windows relay forwards to WSL's own address; pass that address, from
`ip -4 -o addr show eth0`, to answer on it alone). It keeps accepting until a connection's first message is a valid
request within LINKTEST_HELLO_S seconds (default 10), dropping silent or malformed ones such as port probes, serves
that one client and exits. No authentication: run it only on the point-to-point link, only while testing.
"""
import os
import socket
import struct
import sys
import time

PORT = 5201
MAX_BULK = 64 << 30          # 'B': bytes the server reads and discards (the client sends 1 GiB, then 4 GiB)
MAX_ECHO = 16 << 20          # 'E': bytes the server echoes back (the client uses at most 64 KiB)
ACCEPT_S = 600               # the server gives up when no valid client arrives in this time
IO_S = 600                   # and when a served client goes quiet this long


def recvall(c, n):
    b = bytearray()
    while len(b) < n:
        p = c.recv(min(n - len(b), 1 << 20))
        if not p:
            raise EOFError
        b += p
    return bytes(b)


def read_request(c):
    """One 9-byte request header -> (op, n), or None on a clean close before it."""
    first = c.recv(9)
    if not first:
        return None
    hdr = first + recvall(c, 9 - len(first))
    return hdr[:1], struct.unpack("!Q", hdr[1:])[0]


def valid(op, n):
    return (op == b"B" and n <= MAX_BULK) or (op == b"E" and n <= MAX_ECHO) or (op == b"Q" and n == 0)


def log(msg):
    print(f"[linktest] {msg}", file=sys.stderr, flush=True)


def handle(c, op, n):
    """Answer one request; False ends the session."""
    if op == b"B":                      # bulk upload: read n bytes, reply ok
        left = n
        while left:
            p = c.recv(min(left, 4 << 20))
            if not p:
                return False
            left -= len(p)
        c.sendall(b"k")
        return True
    if op == b"E":                      # echo: reply with n bytes
        c.sendall(recvall(c, n))
        return True
    return False                        # Q


def serve(host="0.0.0.0", port=PORT, hello_s=None, accept_s=ACCEPT_S, on_listen=None):
    """Serve the first client whose first request is valid, then return."""
    hello_s = float(os.environ.get("LINKTEST_HELLO_S", "10")) if hello_s is None else hello_s
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(8)
    if on_listen is not None:
        on_listen(s.getsockname())
    deadline = time.monotonic() + accept_s
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no valid client within {accept_s:.0f} s")
            s.settimeout(left)
            c, addr = s.accept()
            c.settimeout(hello_s)
            try:
                req = read_request(c)
            except (OSError, EOFError) as e:
                req = e
            if not isinstance(req, tuple) or not valid(*req):
                log(f"dropped {addr[0]}:{addr[1]} (first message {req!r})")
                c.close()
                continue
            break
    finally:
        s.close()
    c.settimeout(IO_S)
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while req is not None and valid(*req) and handle(c, *req):
            req = read_request(c)
        if req is not None and not valid(*req):
            log(f"stopped at an invalid request {req!r}")
    except (OSError, EOFError) as e:
        log(f"client gone: {e!r}")
    finally:
        c.close()


def pct(lat, q):
    """The q-th percentile of sorted latencies (lat[1000], lat[1800], lat[1980] for 2000 rounds, as before)."""
    return lat[min(len(lat) - 1, len(lat) * q // 100)]


def client(host, port=PORT, bulk=(1 << 30, 4 << 30), sizes=(64, 16384, 65536), rounds=2000):
    c = socket.create_connection((host, port), timeout=60); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf = b"\0" * (4 << 20)
    for total in bulk:
        c.sendall(b"B" + struct.pack("!Q", total)); t0 = time.perf_counter(); left = total
        while left:
            k = min(left, len(buf)); c.sendall(buf[:k]); left -= k
        recvall(c, 1); dt = time.perf_counter() - t0
        print(f"bulk {total >> 30} GiB: {dt:.2f} s = {total / dt / 1e9 * 8:.2f} Gbit/s ({total / dt / 2**20:.0f} MiB/s)")
    for size in sizes:
        msg = b"x" * size; lat = []
        for _ in range(rounds):
            t0 = time.perf_counter(); c.sendall(b"E" + struct.pack("!Q", size) + msg); recvall(c, size); lat.append(time.perf_counter() - t0)
        lat.sort()
        print(f"round trip {size:>6} B: p50 {pct(lat, 50)*1e3:.3f} ms  p90 {pct(lat, 90)*1e3:.3f} ms  p99 {pct(lat, 99)*1e3:.3f} ms")
    c.sendall(b"Q" + struct.pack("!Q", 0)); c.close()


def main(argv):
    if argv[:1] == ["server"] and len(argv) <= 3:
        serve(argv[2] if len(argv) > 2 else "0.0.0.0", int(argv[1]) if len(argv) > 1 else PORT)
        return 0
    if argv[:1] == ["client"] and 2 <= len(argv) <= 3:
        client(argv[1], int(argv[2]) if len(argv) > 2 else PORT)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
