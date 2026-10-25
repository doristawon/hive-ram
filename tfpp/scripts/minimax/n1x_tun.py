#!/usr/bin/env python3
"""A point-to-point IP link between two WSL2 hosts that can open only one TCP connection, in one direction, and
whose network configuration must not change.

The process forks. The parent stays in the host's network namespace and owns the TCP side: the server end keeps
the listening socket and accepts, the client end dials (with retries). The child moves into a fresh, unprivileged
user + network namespace, creates a TUN device there and pumps packets over whatever connection the parent hands
it (over a Unix socket pair, SCM_RIGHTS). When the connection drops (the userland relay on the way may cut it), the
child asks for a new one and keeps the device, so TCP sessions inside the namespace ride out the gap. A heartbeat
frame every few seconds keeps the relay from seeing an idle connection.

Neither host's interfaces, routes, firewall or WSL settings are touched: the device exists only inside the
namespace and disappears with the child. Programs that should talk across the link join the namespace with

    nsenter --preserve-credentials -U -n -t $(cat PIDFILE) <command>

  n1x_tun.py server [BIND:]PORT ADDR/PREFIX    e.g. server 5201 10.99.0.1/30  (the end the relay forwards to)
  n1x_tun.py client HOST:PORT ADDR/PREFIX      e.g. client 10.77.0.1:5201 10.99.0.2/30

Options (environment): N1X_TUN_DEV (n1x0), N1X_TUN_MTU (16000), N1X_TUN_PIDFILE (~/n1x-cluster/n1x_tun.pid; it
holds the namespace child's pid), N1X_TUN_BEAT_S (5).
"""

import fcntl
import os
import select
import socket
import struct
import subprocess
import sys
import threading
import time

TUNSETIFF = 0x400454CA
IFF_TUN, IFF_NO_PI = 0x0001, 0x1000
HDR = struct.Struct("!I")          # frame length; 0 = heartbeat


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} n1x_tun[{os.getpid()}]: {msg}", flush=True)


def tune(s):
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        s.setsockopt(socket.SOL_SOCKET, opt, 8 << 20)


def broker(mode, where, ctl):
    """Parent, host namespace: hand the child one connected socket each time it asks (one byte on ctl)."""
    host, _, port = where.rpartition(":")
    ls = None
    if mode == "server":
        ls = socket.socket()
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind((host or "0.0.0.0", int(port)))
        ls.listen(4)
        log(f"listening on {host or '0.0.0.0'}:{port}")
    while True:
        if not ctl.recv(1):
            return                                         # the child is gone
        while True:
            try:
                if ls is not None:
                    s, peer = ls.accept()
                else:
                    s = socket.create_connection((host, int(port)), timeout=10)
                    s.settimeout(None)
                    peer = (host, port)
                tune(s)
                break
            except OSError as exc:
                log(f"connect failed ({exc!r}); retrying")
                time.sleep(1)
        log(f"connected to {peer}")
        socket.send_fds(ctl, [b"c"], [s.fileno()])
        s.close()


def enter_namespace():
    uid, gid = os.getuid(), os.getgid()
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)
    with open("/proc/self/setgroups", "w") as f:
        f.write("deny")
    with open("/proc/self/uid_map", "w") as f:
        f.write(f"0 {uid} 1")
    with open("/proc/self/gid_map", "w") as f:
        f.write(f"0 {gid} 1")


def make_tun(dev, addr, mtu):
    fd = os.open("/dev/net/tun", os.O_RDWR)
    fcntl.ioctl(fd, TUNSETIFF, struct.pack("16sH", dev.encode(), IFF_TUN | IFF_NO_PI))
    for cmd in (["ip", "link", "set", "lo", "up"],
                ["ip", "addr", "add", addr, "dev", dev],
                ["ip", "link", "set", dev, "mtu", str(mtu), "up"]):
        subprocess.run(cmd, check=True)
    return fd


def recv_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def pump(ctl, dev, addr, mtu, pidfile, beat):
    """Child, own namespace: the TUN device, and packets over the current connection."""
    enter_namespace()
    fd = make_tun(dev, addr, mtu)
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
    log(f"{dev} {addr} mtu {mtu} up in this namespace")
    lock = threading.Lock()
    send_lock = threading.Lock()                           # one frame at a time on the wire
    cur = {"s": None, "gen": 0}
    ready = threading.Condition(lock)

    def new_conn():
        ctl.send(b"n")
        _, fds, _, _ = socket.recv_fds(ctl, 1, 1)
        s = socket.socket(fileno=fds[0])
        with lock:
            cur["s"], cur["gen"] = s, cur["gen"] + 1
            ready.notify_all()
        return s

    def send(frame):
        with lock:
            s = cur["s"]
        if s is None:
            return                                         # reconnecting: drop, TCP above retransmits
        try:
            with send_lock:
                s.sendall(frame)
        except OSError:
            pass                                           # the reader notices and reconnects

    def tun_to_sock():
        while True:
            pkt = os.read(fd, 65535)
            send(HDR.pack(len(pkt)) + pkt)

    def heartbeat():
        while True:
            time.sleep(beat)
            send(HDR.pack(0))

    threading.Thread(target=tun_to_sock, daemon=True).start()
    threading.Thread(target=heartbeat, daemon=True).start()
    s = new_conn()
    while True:
        try:
            while True:
                (n,) = HDR.unpack(recv_exact(s, HDR.size))
                if n:
                    os.write(fd, recv_exact(s, n))
        except Exception as exc:
            log(f"link lost ({exc!r}); reconnecting")
            with lock:
                cur["s"] = None
            try:
                s.close()
            except OSError:
                pass
            s = new_conn()


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("server", "client"):
        sys.exit(__doc__)
    mode, where, addr = sys.argv[1:]
    dev = os.environ.get("N1X_TUN_DEV", "n1x0")
    mtu = int(os.environ.get("N1X_TUN_MTU", "16000"))
    beat = float(os.environ.get("N1X_TUN_BEAT_S", "5"))
    pidfile = os.environ.get("N1X_TUN_PIDFILE", os.path.expanduser("~/n1x-cluster/n1x_tun.pid"))
    parent_end, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    pid = os.fork()
    if pid == 0:
        parent_end.close()
        try:
            pump(child_end, dev, addr, mtu, pidfile, beat)
        finally:
            os._exit(1)
    child_end.close()
    try:
        broker(mode, where, parent_end)
    finally:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
        try:
            os.unlink(pidfile)
        except OSError:
            pass


if __name__ == "__main__":
    main()
