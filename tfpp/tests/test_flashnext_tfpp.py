# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Flash Next tfpp (tfpp/proto.py, wire.py, driver.py's protocol, tfpp_run.py, compare.py,
rank_bench.py): no GPU, no TensorFold, no torch needed.

  cd tfpp && python -m pytest -q tests

driver.py runs here against a fake engine (stand-ins for torch and TensorFold's decode/forward/MTP modules) over real
localhost sockets: the leader's tokens must equal a greedy reference, every rank must commit the same rows, and a
follower must wait idle (no KeepWarm, no timeout) exactly between requests.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True

import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
T = HERE.parent                                           # the tfpp checkout (tfpp/tests -> tfpp)
if str(T) not in sys.path:
    sys.path.insert(0, str(T))

from tfpp import proto, wire  # noqa: E402
from tfpp.proto import LinkError, ProtocolError, RemoteError  # noqa: E402


@pytest.fixture(autouse=True)
def _no_secret(monkeypatch):
    monkeypatch.delenv("TFPP_SECRET", raising=False)


# -- proto: command line ------------------------------------------------------------------------------------------
def test_parse_layers():
    assert proto.parse_layers("0:32") == (0, 32)
    assert proto.parse_layers("32:48") == (32, 48)
    for bad in ("", "32", "32:32", "48:32", "-1:5", "a:b", "1:2:3", "0-32"):
        with pytest.raises(ValueError):
            proto.parse_layers(bad)


def test_parse_peer():
    assert proto.parse_peer("10.77.0.1:5201") == ("10.77.0.1", 5201)
    assert proto.parse_peer("[::1]:5201") == ("::1", 5201)
    for bad in ("", "host", ":5201", "host:", "host:x", "host:0", "host:70000"):
        with pytest.raises(ValueError):
            proto.parse_peer(bad)


def test_check_role_range():
    proto.check_role_range("leader", 0, 32, 48)
    proto.check_role_range("middle", 16, 32, 48)
    proto.check_role_range("last", 32, 48, 48)
    for role, a, b in (("leader", 4, 32), ("last", 32, 47), ("middle", 40, 52), ("middle", 32, 32),
                       ("middle", 48, 32), ("middle", -1, 5)):
        with pytest.raises(ValueError):
            proto.check_role_range(role, a, b, 48)


def test_config_fingerprint(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    (a / "config.json").write_text('{"text_config": {"num_hidden_layers": 48, "vocab_size": 1000}}', encoding="utf-8")
    (b / "config.json").write_text('{\n  "text_config": {\n    "vocab_size": 1000,\n    "num_hidden_layers": 48\n  }\n}\n',
                                   encoding="utf-8")
    (c / "config.json").write_text('{"text_config": {"num_hidden_layers": 40, "vocab_size": 1000}}', encoding="utf-8")
    fa, fb, fc = (proto.config_fingerprint(d) for d in (a, b, c))
    assert fa == fb and fa != fc and len(fa) == 64
    assert proto.config_fingerprint(tmp_path / "missing") is None


# -- proto: the chain (HELLO) -------------------------------------------------------------------------------------
def rec(role, first, end, *, total=48, capacity=16384, rows=8, mtp=False, **kw):
    r = {"proto": proto.PROTO, "role": role, "first": first, "end": end, "total": total, "capacity": capacity,
         "prefill_rows": 2048, "rows": rows, "depth": 0, "mtp": mtp, "around_one": True, "config": "c0",
         "tensorfold": "0.6.1"}
    r.update(kw)
    return r


def test_check_chain_accepts_contiguous_splits():
    assert proto.check_chain([rec("leader", 0, 32), rec("last", 32, 48)], 0) == []
    assert proto.check_chain([rec("leader", 0, 16), rec("middle", 16, 32), rec("last", 32, 48)], 0) == []
    assert proto.check_chain([rec("leader", 0, 32, rows=8), rec("last", 32, 48, rows=8, mtp=True)], 6) == []
    assert proto.check_chain([rec("leader", 0, 32), rec("last", 32, 48, capacity=32768)], 0) == []


@pytest.mark.parametrize("ranks, depth, needle", [
    ([rec("leader", 0, 32), rec("last", 33, 48)], 0, "a gap"),
    ([rec("leader", 0, 33), rec("last", 32, 48)], 0, "an overlap"),
    ([rec("leader", 4, 32), rec("last", 32, 48)], 0, "starts at layer 4"),
    ([rec("leader", 0, 32), rec("last", 32, 40)], 0, "ends at layer 40"),
    ([rec("leader", 0, 48), rec("last", 0, 48)], 0, "an overlap"),              # both left at --layers 0:48
    ([rec("leader", 0, 32), rec("last", 32, 48, around_one=False)], 0, "around_one"),
    ([rec("leader", 0, 32), rec("last", 32, 48, config="c1")], 0, "config"),
    ([rec("leader", 0, 32), rec("last", 32, 48, tensorfold="1.0")], 0, "tensorfold"),
    ([rec("leader", 0, 32), rec("last", 32, 48, capacity=8192)], 0, "cache capacity"),
    ([rec("leader", 0, 32, rows=9), rec("last", 32, 48, rows=8, mtp=True)], 8, "needs 9"),
    ([rec("leader", 0, 32), rec("last", 32, 48, mtp=False)], 6, "no MTP head"),
    ([rec("leader", 0, 32), rec("middle", 32, 48)], 0, "not at a last rank"),
    ([rec("leader", 0, 16), rec("last", 16, 32), rec("last", 32, 48)], 0, "must be a middle rank"),
    ([rec("last", 0, 32), rec("last", 32, 48)], 0, "must be the leader"),
    ([rec("leader", 0, 48)], 0, "not at a last rank"),
])
def test_check_chain_refuses(ranks, depth, needle):
    problems = proto.check_chain(ranks, depth)
    assert any(needle in p for p in problems), problems


# -- proto: frames ------------------------------------------------------------------------------------------------
LIM = proto.Limits(vocab=1000, rows=8, prefill_rows=16, width=8)


def frame(op, **kw):
    h = {"op": op}
    if op in ("prefill", "step"):
        h.update(tokens=[1, 2, 3], depth=3, remaining=10, conf=0.6)
        if op == "prefill":
            h.update(next=[2, 3], final=True)
    h.update(kw)
    return h


def test_check_frame_accepts_what_the_leader_sends():
    assert proto.check_frame({"op": "hello", "proto": proto.PROTO}, None, LIM, hello_done=False) == "hello"
    assert proto.check_frame(frame("reset"), None, LIM, hello_done=True) == "reset"
    assert proto.check_frame(frame("quit"), None, LIM, hello_done=True) == "quit"
    assert proto.check_frame(frame("prefill"), (3, 8), LIM, hello_done=True) == "prefill"
    assert proto.check_frame(frame("prefill", tokens=list(range(16)), next=list(range(1, 17)), final=False),
                             (16, 8), LIM, hello_done=True) == "prefill"
    assert proto.check_frame(frame("step", remaining=0, conf=0), (3, 8), LIM, hello_done=True) == "step"
    assert proto.check_frame({"op": "error", "where": "x", "msg": "y"}, None, LIM, hello_done=False) == "error"


@pytest.mark.parametrize("hdr, shape, hello_done, needle", [
    ({"op": "exec"}, None, True, "unknown op"),
    ({"tokens": [1]}, None, True, "unknown op"),
    (frame("reset"), None, False, "before hello"),
    ({"op": "hello", "proto": proto.PROTO}, None, True, "second hello"),
    ({"op": "hello", "proto": "tfpp-flashnext/1"}, None, False, "protocol"),
    (frame("reset"), (1, 8), True, "with a tensor"),
    (frame("step", tokens=[]), (0, 8), True, "0 ids"),
    (frame("step", tokens=list(range(9))), (9, 8), True, "9 ids"),
    (frame("prefill", tokens=list(range(17)), next=[]), (17, 8), True, "17 ids"),
    (frame("step", tokens=[1, 1000]), (2, 8), True, "token ids"),
    (frame("step", tokens=[-1]), (1, 8), True, "token ids"),
    (frame("step", tokens=[True]), (1, 8), True, "token ids"),
    (frame("step", tokens=[1.0]), (1, 8), True, "token ids"),
    (frame("step", tokens="123"), (3, 8), True, "not a list"),
    (frame("step"), None, True, "residual rows"),
    (frame("step"), (1, 8), True, "residual rows"),
    (frame("step"), (3, 9), True, "residual rows"),
    (frame("step", depth=8), (3, 8), True, "depth"),
    (frame("step", depth=-1), (3, 8), True, "depth"),
    (frame("step", remaining=-1), (3, 8), True, "remaining"),
    (frame("step", remaining="5"), (3, 8), True, "remaining"),
    (frame("step", conf="0.6"), (3, 8), True, "conf"),
    (frame("step", conf=None), (3, 8), True, "conf"),
    (frame("prefill", next=[1, 2, 3, 4]), (3, 8), True, "4 ids"),
    (frame("prefill", next=[5000]), (3, 8), True, "token ids"),
    (frame("prefill", final=1), (3, 8), True, "final"),
])
def test_check_frame_refuses(hdr, shape, hello_done, needle):
    with pytest.raises(ProtocolError, match=needle):
        proto.check_frame(hdr, shape, LIM, hello_done=hello_done)


def test_check_reply():
    assert proto.check_reply({"ok": 1}, "reset", vocab=10)
    assert proto.check_reply({"first": 3, "drafts": [1, 2], "done": False}, "prefill", vocab=10, depth=2)
    assert proto.check_reply({"keep": 2, "sampled": [1, 2], "drafts": []}, "step", vocab=10, rows=3, depth=0)
    with pytest.raises(RemoteError, match="layers 32:48: boom"):
        proto.check_reply({"op": "error", "where": "layers 32:48", "msg": "boom"}, "step", vocab=10, rows=1)
    for r, kind, kw in (({"ok": 0}, "reset", {}),
                        ({"first": 10, "drafts": []}, "prefill", {}),
                        ({"first": 3, "drafts": [1, 2, 3]}, "prefill", {"depth": 2}),
                        ({"keep": 0, "sampled": [], "drafts": []}, "step", {"rows": 3}),
                        ({"keep": 4, "sampled": [1, 2, 3, 4], "drafts": []}, "step", {"rows": 3}),
                        ({"keep": 2, "sampled": [1], "drafts": []}, "step", {"rows": 3}),
                        ({"keep": 1, "sampled": [1], "drafts": [], "done": "yes"}, "step", {"rows": 3}),
                        ({"op": "hello", "ranks": "x"}, "hello", {})):
        with pytest.raises(ProtocolError):
            proto.check_reply(r, kind, vocab=10, **kw)


def test_check_request():
    ok = dict(vocab=100, capacity=64, rows=8)
    proto.check_request([1, 2, 3], 10, 7, 0.6, **ok)
    proto.check_request(list(range(64)), 1000, 0, 0, **ok)        # the count is not capped up front (EOS may come)
    for args, needle in ((([], 10, 0, 0.6), "at least one token"),
                         (([1, 100], 10, 0, 0.6), "integers"),
                         (([1, True], 10, 0, 0.6), "integers"),
                         ((list(range(65)), 10, 0, 0.6), "capacity"),
                         (([1], 0, 0, 0.6), "count"),
                         (([1], 10, -1, 0.6), "depth"),
                         (([1], 10, 8, 0.6), "rows"),
                         (([1], 10, 0, "x"), "confidence")):
        with pytest.raises(ValueError, match=needle):
            proto.check_request(*args, **ok)


def test_done_flags_match_the_leaders_loop():
    eos = {7}
    assert proto.done_after_prefill(0, 3, eos) and proto.done_after_prefill(5, 7, eos)
    assert not proto.done_after_prefill(5, 3, eos)
    assert proto.done_after_step(2, [1, 2], eos) and proto.done_after_step(5, [1, 7], eos)
    assert not proto.done_after_step(3, [1, 2], eos)


# -- wire ---------------------------------------------------------------------------------------------------------
def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def tcp_pair(**kw):
    """Two raw Links over localhost TCP (no link handshake)."""

    ls = socket.socket()
    ls.bind(("127.0.0.1", 0))
    ls.listen(1)
    c = socket.create_connection(ls.getsockname())
    s, _ = ls.accept()
    ls.close()
    return wire.Link(c, **kw), wire.Link(s, **kw)


def raw_frame(header) -> bytes:
    raw = json.dumps(header).encode() if not isinstance(header, bytes) else header
    return struct.pack("<I", len(raw)) + raw


def test_sockets_keep_kernel_buffers_and_keepalive():
    a, b = tcp_pair()
    assert a.sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
    assert "socket buffers" in a.describe()
    a.close(), b.close()


def test_header_round_trip():
    a, b = tcp_pair(io_timeout=5)
    a.send({"op": "reset"})
    a.send({"op": "step", "tokens": [1, 2]})
    assert b.recv() == ({"op": "reset"}, None)
    assert b.recv(timeout=None) == ({"op": "step", "tokens": [1, 2]}, None)
    with pytest.raises(ValueError):
        a.send({"op": "x", "pad": "y" * (wire.MAX_HEADER + 1)})
    a.close(), b.close()


@pytest.mark.parametrize("data, needle", [
    (b"\xff\xff\xff\xff", "frame header"),          # a 4 GiB header length: refused before any allocation
    (b"\x00\x00\x00\x00", "frame header"),
    (raw_frame(b"not json"), "not JSON"),
    (raw_frame([1, 2]), "JSON object"),
    (b"B" + struct.pack("!Q", 1 << 20), "not JSON"),  # linktest.py's opening bytes
])
def test_recv_refuses_malformed_frames(data, needle):
    a, b = tcp_pair(io_timeout=2)
    a.sock.sendall(data + b"\0" * 80)
    with pytest.raises(ProtocolError, match=needle):
        b.recv()
    a.close(), b.close()


@pytest.mark.parametrize("header, max_payload, needle", [
    ({"op": "step", "nbytes": 2 ** 34, "shape": [2 ** 33]}, 1 << 20, "at most"),
    ({"op": "step", "nbytes": 16, "shape": [2, 4]}, 0, "at most"),
    ({"op": "step", "nbytes": 16, "shape": [2, 5]}, 1 << 20, "bf16 payload"),
    ({"op": "step", "nbytes": 16, "shape": "2x4"}, 1 << 20, "shape"),
    ({"op": "step", "nbytes": 16, "shape": [-2, -4]}, 1 << 20, "shape"),
    ({"op": "step", "nbytes": "16", "shape": [2, 4]}, 1 << 20, "payload of"),
    ({"op": "step", "nbytes": -16, "shape": [2, 4]}, 1 << 20, "payload of"),
])
def test_recv_checks_the_payload_before_allocating(header, max_payload, needle):
    a, b = tcp_pair(io_timeout=2, max_payload=max_payload)
    a.sock.sendall(raw_frame(header))
    with pytest.raises(ProtocolError, match=needle):
        b.recv()
    a.close(), b.close()
    assert wire.check_payload({"nbytes": 16, "shape": [2, 4]}, 16) == [2, 4]


def test_recv_timeouts_and_close():
    a, b = tcp_pair(io_timeout=0.3)
    with pytest.raises(LinkError, match="nothing received"):
        b.recv()
    with pytest.raises(LinkError, match="nothing received"):
        b.recv(timeout=0.2)
    a.sock.sendall(struct.pack("<I", 100) + b"{")            # a frame that stops half-way
    t = time.monotonic()
    with pytest.raises(LinkError, match="middle of a frame"):
        b.recv(timeout=None)
    assert time.monotonic() - t < 5
    a.close()
    with pytest.raises(LinkError, match="peer closed"):
        b.recv()
    b.close()


def test_fail_sends_an_error_frame():
    a, b = tcp_pair(io_timeout=5)
    threading.Thread(target=lambda: (time.sleep(0.2), b.close()), daemon=True).start()
    a.fail("layers 32:48", "RuntimeError: boom", drain_s=0)
    hdr, _ = b.recv()
    assert hdr == {"op": "error", "where": "layers 32:48", "msg": "RuntimeError: boom"}
    a.close()


class Listener(threading.Thread):
    def __init__(self, port, **kw):
        super().__init__(daemon=True)
        self.port, self.kw, self.link, self.exc = port, kw, None, None
        self.start()

    def run(self):
        try:
            self.link = wire.Link.listen(self.port, "127.0.0.1", **self.kw)
        except BaseException as exc:  # noqa: BLE001
            self.exc = exc


def test_link_handshake_and_any_start_order():
    port = free_port()
    box = {}

    def late():
        time.sleep(1.5)
        box["l"] = Listener(port, io_timeout=5)

    threading.Thread(target=late, daemon=True).start()
    t = time.monotonic()
    down = wire.Link.connect("127.0.0.1", port, connect_timeout=20, io_timeout=5)
    assert 1.0 < time.monotonic() - t < 15
    lst = box["l"]
    lst.join(5)
    assert lst.exc is None and lst.link is not None
    down.send({"op": "hello", "proto": proto.PROTO})
    assert lst.link.recv()[0]["op"] == "hello"
    down.close(), lst.link.close()


def test_connect_gives_up_on_a_silent_accept():
    """A relay that accepts with nobody behind it: not linked, retried, then a clear error (no hang)."""

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    held = []
    stop = threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                held.append(srv.accept()[0])
            except OSError:
                pass

    threading.Thread(target=accept, daemon=True).start()
    t = time.monotonic()
    with pytest.raises(LinkError, match="could not link"):
        wire.Link.connect("127.0.0.1", srv.getsockname()[1], connect_timeout=1.5, handshake_s=0.5)
    assert time.monotonic() - t < 10
    stop.set()
    srv.close()
    for s in held:
        s.close()


def test_connect_refuses_another_protocol_at_once():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        c, _ = srv.accept()
        c.sendall(raw_frame({"op": "ready", "proto": "something-else/9", "nonce": "0" * 32}))
        time.sleep(1)
        c.close()

    threading.Thread(target=serve, daemon=True).start()
    t = time.monotonic()
    with pytest.raises(ProtocolError, match="not a tfpp"):
        wire.Link.connect("127.0.0.1", srv.getsockname()[1], connect_timeout=30)
    assert time.monotonic() - t < 5
    srv.close()


def test_listener_drops_strangers_and_keeps_listening():
    port = free_port()
    lst = Listener(port, io_timeout=5)
    time.sleep(0.3)
    for data in (b"B" + struct.pack("!Q", 1 << 20), b"GET / HTTP/1.0\r\n\r\n", b""):
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(data)
        time.sleep(0.2)
        s.close()
    assert lst.is_alive() and lst.link is None
    down = wire.Link.connect("127.0.0.1", port, connect_timeout=10, io_timeout=5)
    lst.join(5)
    assert lst.link is not None
    down.close(), lst.link.close()


def test_shared_secret():
    port = free_port()
    lst = Listener(port, io_timeout=5, secret=b"s3")
    down = wire.Link.connect("127.0.0.1", port, connect_timeout=10, secret=b"s3")
    lst.join(5)
    assert lst.link is not None
    down.close(), lst.link.close()

    port = free_port()
    lst = Listener(port, io_timeout=5, secret=b"s3")
    for wrong in (b"", b"other"):                             # refused at once, no retry; the listener stays
        with pytest.raises(ProtocolError, match="refused"):
            wire.Link.connect("127.0.0.1", port, connect_timeout=30, secret=wrong)
    assert lst.is_alive()
    down = wire.Link.connect("127.0.0.1", port, connect_timeout=10, secret=b"s3")
    lst.join(5)
    assert lst.link is not None
    down.close(), lst.link.close()

    port = free_port()
    lst = Listener(port, io_timeout=5, secret=b"")
    with pytest.raises(ProtocolError, match="did not prove"):
        wire.Link.connect("127.0.0.1", port, connect_timeout=30, secret=b"s3")
    lst.join(5)
    lst.link.close()


# -- compare.py, tfpp_run.py, rank_bench.py -----------------------------------------------------------------------
def result(prompt, tokens, text="颱風夜"):
    return {"prompt": prompt, "tokens": tokens, "text": text, "decode_tps": 30.0, "prefill_tps": 900.0,
            "accepted": 1, "drafted": 2}


def run_compare(tmp_path, ref, got):
    pa, pb = tmp_path / "ref.json", tmp_path / "pp.json"
    pa.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    pb.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="0", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(T / "compare.py"), str(pa), str(pb)], capture_output=True, text=True,
                          env=env, timeout=60)


def test_compare_exit_codes(tmp_path):
    ref = [result(0, [1, 2, 3]), result(1, [4, 5])]
    p = run_compare(tmp_path, ref, ref)
    assert p.returncode == 0 and "2/2 identical" in p.stdout, p.stderr
    p = run_compare(tmp_path, ref, [result(0, [1, 2, 3]), result(1, [4, 6])])
    assert p.returncode == 1 and "1/2 identical" in p.stdout
    for got in (ref[:1], ref + [result(2, [7])], [ref[1], ref[0]], []):
        assert run_compare(tmp_path, ref, got).returncode == 2
    assert run_compare(tmp_path, [], []).returncode == 2


def tfpp_run_args(*argv):
    import tfpp_run

    ap = tfpp_run.parser()
    a = ap.parse_args(list(argv))
    tfpp_run.check_args(ap, a)
    return a


def test_tfpp_run_refuses_bad_arguments_before_loading():
    a = tfpp_run_args("--role", "leader", "--model", "M", "--layers", "0:32", "--peer", "10.77.0.1:5201")
    assert (a.first, a.end, a.bind) == (0, 32, os.environ.get("TFPP_BIND", "0.0.0.0"))
    tfpp_run_args("--role", "last", "--model", "M", "--layers", "32:48", "--listen", "5201", "--depth", "6")
    tfpp_run_args("--role", "local", "--model", "M")
    for argv in (("--role", "leader", "--model", "M", "--layers", "0:32"),
                 ("--role", "leader", "--model", "M", "--layers", "0:32", "--peer", "10.77.0.1"),
                 ("--role", "middle", "--model", "M", "--peer", "h:1"),
                 ("--role", "last", "--model", "M", "--layers", "32:32"),
                 ("--role", "last", "--model", "M", "--layers", "a:b"),
                 ("--role", "last", "--model", "M", "--layers", "32:48", "--peer", "h:1"),
                 ("--role", "local", "--model", "M", "--count", "0"),
                 ("--role", "local", "--model", "M", "--depth", "-1"),
                 ("--role", "local", "--model", "M", "--conf", "2")):
        with pytest.raises(SystemExit) as exc:
            tfpp_run_args(*argv)
        assert exc.value.code == 2


def test_tfpp_run_saves_utf8(tmp_path):
    import tfpp_run

    out = tmp_path / "pp.json"
    tfpp_run.save(str(out), [result(0, [1], "停電")])
    assert json.loads(out.read_text(encoding="utf-8"))[0]["text"] == "停電"


def test_rank_bench_modes():
    import rank_bench

    assert rank_bench.parse_modes("sleep@0,gpuwarm@30") == [("sleep", 0.0), ("gpuwarm", 30.0)]
    for bad in ("sleep", "nap@3", "sleep@x", "sleep@-1"):
        with pytest.raises(ValueError):
            rank_bench.parse_modes(bad)


# -- driver.py against a fake engine ------------------------------------------------------------------------------
TF = "tensorfold.families.qwen4_exp.cuda"
FAKED = ["torch", "tensorfold", "tensorfold.families", "tensorfold.families.qwen4_exp", TF, TF + ".decode",
         TF + ".forward", TF + ".mtp", TF + ".graphs", "tfpp.driver", "tfpp.part", "tfpp.warm", "tfpp.wire"]
V, WIDTH = 1000, 8


def nxt(tok: int, pos: int) -> int:
    """The fake model: the greedy token at ``pos`` after ``tok`` (never 997..999)."""

    return (tok * 31 + pos * 7 + 3) % 997


def reference(prompt, count, eos):
    out = [nxt(prompt[-1], len(prompt))]
    while len(out) < count and out[-1] not in eos:
        out.append(nxt(out[-1], len(prompt) + len(out)))
    return out[:count]


def fake_torch(np):
    class FT:
        """A tensor over a numpy array (bf16 as its 16-bit pattern)."""

        def __init__(self, a):
            self.a = a

        shape = property(lambda self: tuple(self.a.shape))

        def contiguous(self):
            return FT(np.ascontiguousarray(self.a))

        def view(self, dtype):
            return FT(self.a.view(np.uint8 if dtype == "uint8" else np.uint16))

        def reshape(self, *shape):
            return FT(self.a.reshape(tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], list) else shape))

        def numel(self):
            return self.a.size

        def copy_(self, other):
            np.copyto(self.a, other.a.reshape(self.a.shape))
            return self

        def numpy(self):
            return self.a

        def to(self, device, non_blocking=False):
            return FT(self.a.copy())

        def clone(self):
            return FT(self.a.copy())

        def __getitem__(self, k):
            return FT(self.a[k])

    class NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    t = types.ModuleType("torch")
    t.uint8, t.bfloat16 = "uint8", "bfloat16"
    t.empty = lambda shape, dtype=None, pin_memory=False: FT(np.zeros(shape, np.uint8 if dtype == "uint8"
                                                                       else np.uint16))
    t.no_grad = NoGrad
    t.cuda = types.SimpleNamespace(synchronize=lambda: None)
    t.FT = FT
    return t


class St:
    def __init__(self):
        self.pos = self.mtp_len = 0

    def set_mtp_len(self, v):
        self.mtp_len = v


class FakeEngine:
    def __init__(self, FT, np, first, end, *, total=16, rows=8, prefill_rows=4, capacity=4096, mtp=True,
                 eos=(), around_one=True, wrong_every=0):
        cfg = types.SimpleNamespace(vocab=V, eos=list(eos), layers=total, streams=2, hidden=WIDTH // 2)
        self.w = types.SimpleNamespace(cfg=cfg, meta={"pp_range": (first, end), "pp_config": "cfg0"},
                                       mtp=object() if mtp else None, around_one=around_one)
        self.st, self.rows, self.prefill_rows, self.capacity, self.graphs = St(), rows, prefill_rows, capacity, None
        self.pbuf = types.SimpleNamespace(streams=FT(np.zeros((prefill_rows, WIDTH), np.uint16)))
        self.buf = types.SimpleNamespace(streams=FT(np.zeros((rows, WIDTH), np.uint16)))
        self.wrong_every, self.fail_at, self.samples = wrong_every, None, 0
        self.FT, self.np = FT, np

    def reset(self):
        self.st.pos = self.st.mtp_len = 0

    def sample(self, logits, positions, sampling):
        self.samples += 1
        if self.fail_at is not None and self.samples >= self.fail_at:
            raise RuntimeError("boom in sample")
        toks = [int(x) for x in logits.a[:, 0]]
        prev = [toks[-1]] if len(positions) == 1 else toks[:len(positions)]
        return [nxt(p, pos) for p, pos in zip(prev, positions)]


@pytest.fixture
def sim(monkeypatch):
    np = pytest.importorskip("numpy")
    import tfpp

    saved = {k: sys.modules.get(k) for k in FAKED}
    attrs = {k: getattr(tfpp, k, None) for k in ("driver", "part", "warm", "wire")}
    torch = fake_torch(np)
    sys.modules["torch"] = torch
    for name in FAKED[1:5]:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    sys.modules["tensorfold"].__version__ = "stub"
    for leaf in ("decode", "forward", "mtp", "graphs"):
        m = types.ModuleType(f"{TF}.{leaf}")
        sys.modules[m.__name__] = m
        setattr(sys.modules[TF], leaf, m)
    sys.modules[TF + ".forward"].commit = None
    sys.modules[TF + ".mtp"].mtp_forward = None
    sys.modules[TF + ".graphs"].Graphs = type("Graphs", (), {})
    monkeypatch.setenv("TFPP_WARM_MS", "0")
    for k in ("driver", "part", "warm", "wire"):
        sys.modules.pop("tfpp." + k, None)
        if hasattr(tfpp, k):
            delattr(tfpp, k)
    try:
        from tfpp import driver
        from tfpp import wire as fwire

        def forward_part(e, tokens, *, prefill, first, last, h_in=None, logits=True):
            R = len(tokens)
            if not first:
                assert h_in is not None and h_in.shape == (R, WIDTH)
            a = np.zeros((R, WIDTH), np.uint16)
            a[:, 0] = tokens
            return torch.FT(a)

        def commit(w, st, buf, R, keep):
            assert 1 <= keep <= R
            st.pos += keep

        def draft(e, streams, next_tokens, position, count, sampling, confidence):
            out, prev = [], next_tokens[-1]
            for j in range(count):
                d = nxt(prev, position + j)
                if e.wrong_every and (position + j) % e.wrong_every == 0:
                    d = (d + 1) % 997
                out.append(d)
                prev = d
            return out

        monkeypatch.setattr(driver, "forward_part", forward_part)
        monkeypatch.setattr(driver, "commit", commit)
        monkeypatch.setattr(driver, "mtp_forward", lambda *a, **k: None)
        monkeypatch.setattr(driver.dec, "draft", draft, raising=False)
        yield types.SimpleNamespace(driver=driver, wire=fwire, np=np, FT=torch.FT)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in attrs.items():
            if v is None:
                if hasattr(tfpp, k):
                    delattr(tfpp, k)
            else:
                setattr(tfpp, k, v)


class Rank(threading.Thread):
    """A follower: links down first (a middle rank), then listens up, then follows; records each up.recv."""

    def __init__(self, s, e, port, down_port=None, depth=0):
        super().__init__(daemon=True)
        self.s, self.e, self.port, self.down_port, self.depth = s, e, port, down_port, depth
        self.exc, self.log = None, []
        self.start()

    def run(self):
        W = self.s.wire.Link
        try:
            down = None
            if self.down_port is not None:
                down = W.connect("127.0.0.1", self.down_port, connect_timeout=20, io_timeout=10)
            up = W.listen(self.port, "127.0.0.1", io_timeout=10)
            recv = up.recv

            def logged(*a, **kw):
                hdr, h = recv(*a, **kw)
                self.log.append((hdr.get("op"), "timeout" in kw and kw["timeout"] is None))
                return hdr, h

            up.recv = logged
            self.s.driver.follow(self.e, up, down, depth=self.depth)
        except BaseException as exc:  # noqa: BLE001
            self.exc = exc


def chain(s, specs, *, depth=0, eos=(), **kw):
    """Start the followers of ``specs`` [(first, end, extra kwargs)], leader first; return (leader, ranks)."""

    ports = [free_port() for _ in specs[1:]]
    ranks = []
    for i in range(len(specs) - 1, 0, -1):
        first, end, extra = specs[i]
        e = FakeEngine(s.FT, s.np, first, end, eos=eos, **{**kw, **extra})
        ranks.insert(0, Rank(s, e, ports[i - 1], ports[i] if i < len(specs) - 1 else None, depth=depth))
    first, end, extra = specs[0]
    e0 = FakeEngine(s.FT, s.np, first, end, eos=eos, **{**kw, **extra})
    link = s.wire.Link.connect("127.0.0.1", ports[0], connect_timeout=20, io_timeout=10)
    return e0, link, ranks


def finish(leader, ranks):
    leader.quit()
    for r in ranks:
        r.join(10)
        assert not r.is_alive()


PROMPTS = [[5, 6, 7], list(range(10, 20)), [42]]


@pytest.mark.parametrize("depth, wrong_every, n", [(0, 0, 2), (3, 0, 2), (3, 4, 2), (6, 3, 3)])
def test_pipeline_matches_the_reference(sim, depth, wrong_every, n):
    specs = [(0, 8, {}), (8, 16, {})] if n == 2 else [(0, 5, {}), (5, 11, {}), (11, 16, {})]
    e0, link, ranks = chain(sim, specs, depth=depth, wrong_every=wrong_every)
    leader = sim.driver.Leader(e0, link, depth=depth, timeout=10)
    assert [r["role"] for r in leader.ranks] == ["leader"] + ["middle"] * (n - 2) + ["last"]
    for prompt in PROMPTS:
        for count in (1, 2, 17):
            got = leader.generate(prompt, count, depth, 0.6)
            assert got["tokens"] == reference(prompt, count, ()), (prompt, count)
            assert {r.e.st.pos for r in ranks} == {e0.st.pos}
            if depth and count > 1 and not wrong_every:
                assert got["accepted"] == got["drafted"] > 0
    finish(leader, ranks)
    for r in ranks:
        assert r.exc is None, r.exc
        ops = [op for op, _ in r.log]
        assert ops[0] == "hello" and ops[-1] == "quit"
        for op, idle in r.log:                            # idle exactly between requests
            assert idle == (op in ("hello", "reset", "quit")), r.log


def test_pipeline_stops_at_eos(sim):
    eos = {reference([5, 6, 7], 30, ())[6]}
    e0, link, ranks = chain(sim, [(0, 8, {}), (8, 16, {})], depth=3, eos=eos)
    leader = sim.driver.Leader(e0, link, depth=3, timeout=10)
    got = leader.generate([5, 6, 7], 30, 3, 0.6)
    assert got["tokens"] == reference([5, 6, 7], 30, eos) and len(got["tokens"]) == 7
    got = leader.generate([42], 5, 3, 0.6)
    assert got["tokens"] == reference([42], 5, eos)
    finish(leader, ranks)
    for op, idle in ranks[0].log:
        assert idle == (op in ("hello", "reset", "quit"))


@pytest.mark.parametrize("specs, depth, needle", [
    ([(0, 8, {}), (9, 16, {})], 0, "a gap"),
    ([(0, 16, {}), (0, 16, {})], 0, "an overlap"),
    ([(0, 8, {}), (8, 16, {"around_one": False})], 0, "around_one"),
    ([(0, 8, {}), (8, 16, {"mtp": False})], 3, "no MTP head"),
    ([(0, 8, {}), (8, 16, {"capacity": 1024})], 0, "cache capacity"),
    ([(0, 5, {}), (5, 12, {}), (11, 16, {})], 0, "an overlap"),
])
def test_hello_refuses_a_broken_chain(sim, specs, depth, needle):
    e0, link, ranks = chain(sim, specs, depth=depth)
    with pytest.raises(ValueError, match=needle):
        sim.driver.Leader(e0, link, depth=depth, timeout=10)
    for r in ranks:
        r.join(10)
        assert isinstance(r.exc, RemoteError) and needle in str(r.exc)
    link.close()


def test_a_failing_rank_reports_through_the_chain(sim):
    e0, link, ranks = chain(sim, [(0, 5, {}), (5, 11, {}), (11, 16, {})], depth=3)
    leader = sim.driver.Leader(e0, link, depth=3, timeout=10)
    assert leader.generate([1, 2, 3], 5, 3, 0.6)["tokens"] == reference([1, 2, 3], 5, ())
    ranks[-1].e.fail_at = ranks[-1].e.samples + 3
    with pytest.raises(RemoteError, match="layers 11:16: RuntimeError: boom in sample"):
        leader.generate(list(range(10, 20)), 40, 3, 0.6)
    assert leader.broken
    with pytest.raises(RuntimeError, match="unknown state"):
        leader.generate([1], 2, 3, 0.6)
    link.close()
    for r in ranks:
        r.join(10)
    assert isinstance(ranks[0].exc, RemoteError) and "boom" in str(ranks[0].exc)
    assert isinstance(ranks[1].exc, RuntimeError) and "boom" in str(ranks[1].exc)


def test_leader_refuses_bad_requests_without_sending(sim):
    e0, link, ranks = chain(sim, [(0, 8, {}), (8, 16, {})], depth=2, capacity=64)
    leader = sim.driver.Leader(e0, link, depth=2, timeout=10)
    for args in (([], 5, 2, 0.6), ([1, V], 5, 2, 0.6), (list(range(65)), 5, 2, 0.6), ([1], 5, 8, 0.6),
                 ([1], 0, 2, 0.6)):
        with pytest.raises(ValueError):
            leader.generate(*args)
    assert not leader.broken
    assert leader.generate([1, 2], 9, 2, 0.6)["tokens"] == reference([1, 2], 9, ())
    finish(leader, ranks)
    assert ranks[0].exc is None
