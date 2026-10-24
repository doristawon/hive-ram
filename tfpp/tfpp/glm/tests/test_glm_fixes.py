"""CPU tests for the fixes made after the v1 hardware runs (IMPLEMENTATION_NOTES.md, "After the hardware runs").

  cd tfpp && python -m pytest -q tfpp/glm/tests/test_glm_fixes.py

No torch or GPU: the socket-level tests import ``wire`` / ``startup`` under a stand-in ``torch`` module (the frames
they exchange carry no tensors). The shell tests need bash with /proc (Linux, WSL; Cygwin's bash too) and are skipped
elsewhere.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

import numpy  # noqa: E402,F401  (before the fixtures' sys.modules snapshots: numpy loads once per process)
import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
GLM = HERE.parent
T = HERE.parents[2]                                     # the tfpp checkout
SCRIPTS = T / "scripts" / "glm"
if str(T) not in sys.path:
    sys.path.insert(0, str(T))

from tfpp.glm import PROTO, capacity, compare, load, proto, v2bench, v2proto  # noqa: E402

CONFIG = json.loads((HERE / "data" / "glm53_config.json").read_text(encoding="utf-8"))
KINDS = ["kda" if k == "linear_attention" else "dsa" for k in CONFIG["text_config"]["layer_types"]]


# -- the socket layer under a stand-in torch --------------------------------------------------------------------------
class _Auto(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (), {}) if name[:1].isupper() and not name.isupper() else mock.MagicMock(name=name)
        setattr(self, name, value)
        return value


@pytest.fixture
def net():
    """``wire`` and ``startup`` imported fresh under a stand-in torch; sys.modules and the package restored after."""

    glm_pkg = sys.modules["tfpp.glm"]
    saved = dict(vars(glm_pkg))
    with mock.patch.dict(sys.modules):
        for k in [k for k in sys.modules if k == "torch" or k.startswith("torch.")
                  or k in ("tfpp.glm.wire", "tfpp.glm.startup")]:
            del sys.modules[k]
        fake = _Auto("torch")
        fake.cuda = mock.MagicMock()
        sys.modules["torch"] = fake
        try:
            yield types.SimpleNamespace(wire=importlib.import_module("tfpp.glm.wire"),
                                        startup=importlib.import_module("tfpp.glm.startup"))
        finally:
            for k in list(vars(glm_pkg)):
                if k not in saved:
                    delattr(glm_pkg, k)
            vars(glm_pkg).update(saved)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def link_bytes(proto_version=PROTO, op=proto.LINK) -> bytes:
    return proto.frame_bytes(proto.make_header(op, -1, {"proto": proto_version, "from": "0:4", "to": "down"}, []))


class Listener:
    """``Link.accept_linked`` in a thread; ``ready()`` waits until it listens (that check is itself a probe that
    connects and closes at once)."""

    def __init__(self, wire, port, **kw):
        self.out: dict = {}
        kw.setdefault("accept_timeout", 20.0)
        kw.setdefault("io_timeout", 10.0)

        def run():
            try:
                self.out["link"] = wire.Link.accept_linked("127.0.0.1", port, "up", "4:8", **kw)
            except BaseException as exc:  # noqa: BLE001
                self.out["err"] = exc

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.port = port

    def ready(self):
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.5).close()
                return
            except OSError:
                time.sleep(0.02)
        raise AssertionError("the listener never listened")

    def result(self, timeout=30.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "accept_linked did not return"
        return self.out


def real_peer(startup, port, out):
    def run():
        try:
            out["link"] = startup.connect_down("127.0.0.1", port, "0:4", connect_timeout=20, io_timeout=10)
        except BaseException as exc:  # noqa: BLE001
            out["err"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def check_linked(net, got, peer_out, peer_thread):
    peer_thread.join(20)
    assert "err" not in got and "err" not in peer_out, (got.get("err"), peer_out.get("err"))
    up, down = got["link"], peer_out["link"]
    assert int(up.peer.rsplit(":", 1)[1]) == down.sock.getsockname()[1], "linked with the stray, not the stage"
    up.send(proto.HELLO, {"stages": []})                    # the link works both ways after the strays
    f = down.recv(timeout=5)
    assert f.op == proto.HELLO and f.hdr["seq"] == 0
    up.close()
    down.close()


def test_accept_drops_a_probe_that_closes_at_once(net):
    lst = Listener(net.wire, free_port(), handshake_s=1.0)
    lst.ready()                                            # a connect-and-close probe
    for _ in range(3):
        socket.create_connection(("127.0.0.1", lst.port), timeout=2).close()
    peer: dict = {}
    t = real_peer(net.startup, lst.port, peer)
    check_linked(net, lst.result(), peer, t)


def test_accept_drops_a_silent_client_and_garbage(net):
    lst = Listener(net.wire, free_port(), handshake_s=0.5)
    lst.ready()
    silent = socket.create_connection(("127.0.0.1", lst.port), timeout=2)          # connects, says nothing
    tls = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
    tls.sendall(b"\x16\x03\x01\x02\x00" + b"\x01" * 64)                            # a TLS ClientHello's first bytes
    other = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
    other.sendall(link_bytes(op="HELLO"))                                           # JSON, but not LINK
    junk = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
    junk.sendall(proto.U32.pack(12) + b"not json at!")                              # a header that does not parse
    t0 = time.monotonic()
    peer: dict = {}
    t = real_peer(net.startup, lst.port, peer)
    check_linked(net, lst.result(), peer, t)
    assert time.monotonic() - t0 < 12, "a stray held the listener past the real peer's own handshake wait"
    for s in (silent, tls, other, junk):
        s.close()


def test_accept_passes_over_a_link_that_closed_and_refuses_another_protocol(net):
    # 5 s, not 1: a loaded machine (CI) must not time out the real peer's handshake; the refusals below stay immediate
    lst = Listener(net.wire, free_port(), handshake_s=5.0)
    lst.ready()
    stale = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
    stale.sendall(link_bytes())                         # a peer that gave up: its LINK waits, its side is closed
    stale.close()
    time.sleep(0.2)
    peer: dict = {}
    t = real_peer(net.startup, lst.port, peer)
    check_linked(net, lst.result(), peer, t)

    lst = Listener(net.wire, free_port(), handshake_s=5.0)
    lst.ready()
    wrong = socket.create_connection(("127.0.0.1", lst.port), timeout=2)
    wrong.sendall(link_bytes(proto_version=PROTO + 98))
    got = lst.result()
    wrong.close()
    assert isinstance(got.get("err"), proto.ProtocolError), got     # a real stage of another version: no waiting


def test_accept_times_out_with_nobody(net):
    t0 = time.monotonic()
    with pytest.raises(net.wire.LinkError, match="no tfpp-glm stage linked"):
        net.wire.Link.accept_linked("127.0.0.1", free_port(), "up", "4:8", accept_timeout=0.5, io_timeout=5)
    assert time.monotonic() - t0 < 5


def test_connect_down_refuses_a_non_tfpp_peer_at_once(net):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    held = []

    def ssh():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            c.sendall(b"SSH-2.0-OpenSSH_9.6\r\n")               # 'SSH-' as a u32 frame size: ~760 MB
            held.append(c)

    threading.Thread(target=ssh, daemon=True).start()
    t0 = time.monotonic()
    with pytest.raises(proto.ProtocolError, match="does not speak"):
        net.startup.connect_down("127.0.0.1", port, "0:4", connect_timeout=30, io_timeout=5)
    assert time.monotonic() - t0 < 5, "retried a peer that will never speak LINK"
    srv.close()
    for c in held:
        c.close()


def test_forward_or_fail_raises_instead_of_sys_exit(net):
    sent = []
    up = types.SimpleNamespace(send_raw=lambda raw, payload: sent.append(raw))
    hdr = proto.make_header(proto.ERROR, 3, {"stage": "leader", "where": "CAP", "msg": "no room"}, [])
    f = types.SimpleNamespace(op=proto.ERROR, hdr=hdr, raw=proto.encode_header(hdr), payload=[])
    with pytest.raises(net.wire.LinkError, match="the leader stopped the chain: CAP: no room"):
        net.startup.forward_or_fail(f, up)
    assert sent, "the ERROR still goes upstream first"
    ok = proto.make_header(proto.GO, 4, {}, [])
    net.startup.forward_or_fail(types.SimpleNamespace(op=proto.GO, hdr=ok, raw=b"x", payload=[]), None)


# -- protocol and tools -----------------------------------------------------------------------------------------------
def test_check_chain_wants_a_kda_layer_in_every_stage():
    proto.check_chain([(0, 6), (6, 8)], 8, KINDS, ())                   # [6, 8): kda + dsa
    with pytest.raises(ValueError, match="no KDA layer"):
        proto.check_chain([(0, 7), (7, 8)], 8, KINDS, ())               # [7, 8): layer 7 alone, DSA only
    with pytest.raises(ValueError, match="no DSA layer"):
        proto.check_chain([(0, 3), (3, 8)], 8, KINDS, ())


def test_mine_knows_an_mlx_embedding():
    for part in ("weight", "scales", "biases"):
        name = f"language_model.model.embed_tokens.{part}"
        assert load.mine(name, 0, 7, is_first=True, is_last=False, mtp=False)
        assert not load.mine(name, 7, 26, is_first=False, is_last=False, mtp=False)
        assert load.mine(name, 26, 45, is_first=False, is_last=True, mtp=False, drafter=True)
    with pytest.raises(ValueError):
        load.mine("language_model.model.embed_tokens.zeros", 0, 7, is_first=True, is_last=False, mtp=False)


def _trace(done=True):
    rows = [{"t": "meta"}, {"t": "req", "n": 1, "prompt_len": 3}, {"t": "chunk", "start": 0, "R": 3, "head": [1, 2]},
            {"t": "step", "pos": 3, "R": 2, "window": [5, 6], "logits": [[1, 1], [2, 2]]}]
    return rows + ([{"t": "done", "n": 1, "tokens": [5, 9]}] if done else [])


def test_compare_never_passes_on_nothing(capsys):
    assert compare.compare(_trace(), _trace(), show=0) == 0
    assert compare.compare([{"t": "meta"}], [{"t": "meta"}], show=0) == 1
    assert "nothing was compared" in capsys.readouterr().out
    assert compare.compare(_trace(done=False), _trace(done=False), show=0) == 1      # both stopped mid-request
    assert "INCOMPLETE" in capsys.readouterr().out
    fail = _trace(done=False) + [{"t": "fail", "n": 1, "error": "ValueError: x"}]
    assert compare.compare(fail, [dict(r) for r in fail], show=0) == 0                 # a failed request did end
    assert compare.compare(_trace(), _trace(done=False), show=0) == 1


def test_agree_names_the_binding_stage():
    class G:
        def __init__(self, per):
            self.per = per

        def needed(self, w):
            return self.per * w

    class P:
        def __init__(self, fitting):
            self.fitting, self.largest, self.settings = fitting, fitting, [1048576, 262144, 1]

    def make_plan(native, requested, explicit, budget, weights, geo):
        return P(budget // geo.per)

    def choose(plan, rows):
        raise ValueError("CUDA startup memory budget cannot fit requested context 262144; estimated largest fitting "
                         "prompt-plus-reply window: 1000 tokens across the ranks.")

    gib = 2 ** 30
    with pytest.raises(ValueError) as exc:
        capacity.agree(native=1048576, requested=262144, explicit=True, budgets=[40 * gib, 1000 * 64, 30 * gib],
                       geos=[G(64), G(64), G(64)], kinds=[(6, 1), (14, 5), (14, 5)], cache_bytes_wanted=0,
                       dense_limit=2051, max_rows=16, dense_capacity=2560, make_plan=make_plan, choose=choose)
    msg = str(exc.value)
    assert msg.startswith("CUDA startup memory budget cannot fit")
    assert "stage 0: budget 40.00 GiB" in msg and "the leader: budget 30.00 GiB" in msg
    assert "fits 1000 tokens" in msg and msg.rstrip(".").endswith("the least: stage 1")


def test_fault_fires_once():
    for mode in ({"kill": True}, {}):
        f = v2proto.Fault("windows:2:stop", armed=True, **mode)
        with mock.patch("os.kill") as kill:
            for _ in range(5):
                f.hit("windows")
            assert kill.call_count == 1, "a stopped stage let go with SIGCONT must carry on"
    k = v2proto.Fault("round.act:2", armed=True)
    k.hit("round.act")
    with pytest.raises(RuntimeError):
        k.hit("round.act")
    k.hit("round.act")                                      # past the N-th: nothing more
    k.hit("round.act")


def test_v2check_fit_shape():
    from tfpp.glm import v2check

    assert v2check.fit_shape((1, 3, 16, 12), 32) == (1, 3, 16, 12)
    for shape in v2check.SHAPES:
        for rows in (16, 24):
            got = v2check.fit_shape(shape, rows)
            assert len(got) == len(shape) and sum(got) <= rows and min(got) >= 1, (shape, rows, got)
    assert len(v2check.fit_shape((8, 8, 8, 8), 16)) == 4


def test_v2check_takes_the_solo_window_before_the_pieces_run():
    """T0e part 3: the solo run's window must be read before the pieces' states (views of the same slots and extents)
    are reset and written, and the pieces' extents are cleared first."""

    fn = next(n for n in ast.walk(ast.parse((GLM / "v2check.py").read_text(encoding="utf-8")))
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    src = ast.unparse(fn)
    sa = src.index("sa = snapshot(serial.forward([Segment(a, t) for a, t in zip(solo_st, wtoks)]))")
    pieces = src.index("pc_st, pc_taps, pc_heads =")
    clear = src.index("clear_extent(st)")
    run = src.index("multi_prefill.prefill_pieces(e, pieces)")
    sb = src.index("sb = snapshot(serial.forward([Segment(a, t) for a, t in zip(pc_st, wtoks)]))")
    assert sa < pieces < clear < run < sb
    assert src.count("snapshot(serial.forward([Segment(a, t) for a, t in zip(solo_st, wtoks)]))") == 1


# -- the lockstep harness comes to rest before it returns -------------------------------------------------------------
class _Stream:
    def __init__(self, tag):
        self.tag = tag


class _Sched:
    class _Flush:
        pass

    def __init__(self):
        self.waiting = queue.Queue()
        self.held, self.loading, self.ready = None, [], []
        self.lanes: dict = {}
        self.decoder = types.SimpleNamespace(live=lambda: len(self.lanes))
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            self._iteration()

    def _iteration(self):
        items = [self.waiting.get()] if not self.lanes else []
        for stream, box in items:
            if stream is not self._Flush:
                self.lanes[stream.tag] = box
        for tag in list(self.lanes):
            time.sleep(0.01)
            self.lanes.pop(tag).put(tag)


def test_lockstep_returns_after_its_last_record():
    sch = _Sched()
    landed = v2bench.Landings()
    recs: list = []
    put = sch.waiting.put

    def client(tag):
        def run():
            box = queue.Queue()
            put((_Stream(tag), box))
            landed.event(tag).set()
            return box.get(timeout=10)
        return run

    lock = v2bench.Lockstep(sch, [v2bench.Req(t, n, client(t)) for t, n in (("a", 0), ("b", 0), ("c", 2))],
                            landed=landed, write=recs.append)
    res = lock.run(timeout=30)
    assert all(ok for ok, _ in res.values())
    assert lock.parked.is_set()
    n = len(recs)
    time.sleep(0.3)
    assert len(recs) == n, "an iter record came after run() returned (a closed trace would miss it)"
    assert recs[-1]["released"] == []


# -- the shell scripts --------------------------------------------------------------------------------------------------
def _bash_with_proc() -> bool:
    if shutil.which("bash") is None:
        return False
    try:
        return subprocess.run(["bash", "-c", "test -r /proc/self/cmdline"], timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_bash = pytest.mark.skipif(not _bash_with_proc(), reason="no bash with /proc")


def bash(script: str, timeout: float = 60.0, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


def bash_path(p: Path) -> str:
    """A path bash can open (Cygwin / MSYS bash on Windows: /c/...; elsewhere the path itself)."""

    if os.name != "nt":
        return str(p)
    return bash(f"cygpath -u '{p}' 2>/dev/null || echo '{p}'").stdout.strip()


def function_lines(path: Path, *names: str) -> str:
    """The one-line definitions of shell functions ``names`` in ``path`` (each must occur exactly once)."""

    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for name in names:
        hits = [l for l in lines if l.startswith(f"{name}() {{")]
        assert len(hits) == 1, (path.name, name, hits)
        out.append(hits[0])
    return "\n".join(out)


@needs_bash
def test_pid_file_names_only_this_roles_stage(tmp_path):
    start = function_lines(SCRIPTS / "start_stage.sh", "is_stage", "stage_pid")
    assert start == function_lines(SCRIPTS / "stop_stage.sh", "is_stage", "stage_pid"), "the two scripts disagree"
    pidf = bash_path(tmp_path / "leader.pid")
    script = f"""set -euo pipefail
ROLE=leader; PIDF={pidf}
{start}
bash -c 'sleep 30; :' -m tfpp.glm.serve --role leader --model x >/dev/null 2>&1 & stage=$!
bash -c 'sleep 30; :' -m tfpp.glm.serve --role follower --model x >/dev/null 2>&1 & other_role=$!
sleep 30 >/dev/null 2>&1 & stranger=$!
sleep 0.5
check() {{ echo "$1" > "$PIDF"; echo "$2=[$(stage_pid)]"; }}
check $stage stage; check $other_role other_role; check $stranger stranger; check 999999 dead; check junk junk
rm -f "$PIDF"; echo "missing=[$(stage_pid)]"
echo "want=$stage"
kill $stage $other_role $stranger
"""
    out = bash(script).stdout
    want = next(l for l in out.splitlines() if l.startswith("want=")).split("=")[1]
    assert f"stage=[{want}]" in out, out
    for case in ("other_role", "stranger", "dead", "junk", "missing"):
        assert f"{case}=[]" in out, out


@needs_bash
def test_stop_stage_leaves_a_stranger_alone_and_survives_no_logs(tmp_path):
    conf, state = tmp_path / "conf", tmp_path / "state"
    conf.mkdir()
    state.mkdir()
    (conf / "node.env").write_bytes(b"ROLE=leader\nLAYERS=0:8\n")
    (conf / "cluster.env").write_bytes(b"TF_GLM_DENSE=q4\n")                # no HTTP_PORT: no health check
    env = {"TFPP_GLM_CONF": bash_path(conf), "TFPP_GLM_STATE": bash_path(state), "STOP_TIMEOUT": "3"}
    stop = bash_path(SCRIPTS / "stop_stage.sh")
    pidf = bash_path(state / "leader.pid")
    script = f"""sleep 30 >/dev/null 2>&1 & stranger=$!
echo $stranger > {pidf}
bash {stop}; echo "rc=$?"
kill -0 $stranger 2>/dev/null && echo stranger-alive
[[ -f {pidf} ]] || echo pidfile-gone
bash -c 'sleep 30; :' -m tfpp.glm.serve --role leader --model x >/dev/null 2>&1 & stage=$!
sleep 0.3; echo $stage > {pidf}
bash {stop}; echo "rc2=$?"
sleep 0.3; kill -0 $stage 2>/dev/null || echo stage-stopped
kill $stranger 2>/dev/null; true
"""
    done = bash(script, env=env)
    out = done.stdout
    assert "rc=0" in out and "stranger-alive" in out and "pidfile-gone" in out, out + done.stderr
    assert "rc2=0" in out and "stage-stopped" in out, out + done.stderr


@needs_bash
def test_start_stage_extension_probe_survives_a_missing_folder(tmp_path):
    line = next(l for l in (SCRIPTS / "start_stage.sh").read_text(encoding="utf-8").splitlines()
                if l.startswith("old_rt=$(find "))
    script = f"""set -euo pipefail
TORCH_EXTENSIONS_DIR={bash_path(tmp_path / "missing" / "ext-cu13")}
{line}
echo "reached [$old_rt]"
"""
    done = bash(script)
    assert done.returncode == 0 and "reached []" in done.stdout, (done.stdout, done.stderr)


@needs_bash
def test_run_v2_fault_case_judges_every_follower():
    text = (SCRIPTS / "run_v2.sh").read_text(encoding="utf-8")
    body = text[text.index("fault_case() {"):]
    body = body[:body.index("\n}\n") + 3]
    stubs = """set -uo pipefail
FAILS=0; OUT=/tmp; SPLITS=("0:4 4:8")
log() { echo "LOG $*"; }
ref() { :; }; judge() { :; }
run_chain() { FRC=($FAKE_FRC); RUN_PIDS=(); PIDS=(); return $FAKE_RC; }
"""
    for frc, rc, extra, want in (("75", 75, "", "PASS"), ("137", 75, "", "FAIL"), ("137", 75, " 137", "PASS"),
                                 ("75", 0, "", "FAIL"), ("", 75, "", "FAIL")):
        script = stubs + body + f'\nfault_case case{extra}\necho "FAILS=$FAILS"\n'
        out = bash(script, env={"FAKE_FRC": frc, "FAKE_RC": str(rc)}).stdout
        verdict = "PASS" if "T1-v2-fault case: PASS" in out else "FAIL"
        assert verdict == want, (frc, rc, extra, out)
