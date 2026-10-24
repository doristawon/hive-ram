"""CPU-only tests for the link tools: linktest.py, the Windows relay and link-setup scripts, scripts/flashnext/env.sh.

Run from tfpp/:  python -m pytest -q -p no:cacheprovider scripts/tests
The PowerShell tests need Windows PowerShell (they skip elsewhere); they never run the setup scripts' bodies (those
change network settings and need admin), only parse them and call their helper functions. The relay test runs
relay_user.ps1 on loopback (127.0.0.1 -> 127.0.0.2), no admin, no system changes.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import queue
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

TFPP = Path(__file__).resolve().parents[2]          # tfpp/
ROOT = TFPP.parent                                   # the checkout: README.md, *.ps1, linktest.py
PS1 = sorted(p.name for p in ROOT.glob("*.ps1"))  # the Windows-side scripts at the checkout's top level
ENV_SH = TFPP / "scripts" / "flashnext" / "env.sh"
POWERSHELL = shutil.which("powershell") if os.name == "nt" else None


def _need(path: Path) -> Path:
    if not path.is_file():
        pytest.skip(f"{path.name} is not in this checkout")
    return path


# ---- file hygiene: these files run on Windows PowerShell 5.1 and in WSL/Linux --------------------------------------

@pytest.mark.parametrize("name", PS1 + ["linktest.py"])
def test_root_tools_are_ascii_lf_without_bom(name):
    data = _need(ROOT / name).read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM"
    assert b"\r" not in data, "CR line endings"
    assert all(b < 128 for b in data), "non-ASCII bytes (Windows PowerShell 5.1 reads BOM-less files as ANSI)"


@pytest.mark.parametrize("name", PS1 + ["linktest.py", "tfpp/scripts/flashnext/env.sh"])
def test_no_absolute_personal_paths(name):
    import re
    text = _need(ROOT / name).read_text(encoding="utf-8")
    for pat in (r"[A-Za-z]:\\Users\\", r"/home/[a-z]", r"/mnt/[a-z]/", r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+"):
        assert not re.search(pat, text), (name, pat)


# ---- linktest.py ------------------------------------------------------------------------------------------------------

def _linktest():
    path = _need(ROOT / "linktest.py")
    spec = importlib.util.spec_from_file_location("linktest_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                   # no __pycache__ next to the checkout's linktest.py
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def _start_server(lt, **kw):
    got = queue.Queue()
    err = []

    def run():
        try:
            lt.serve("127.0.0.1", 0, on_listen=got.put, **kw)
        except BaseException as e:                   # surfaced by the test
            err.append(e)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    port = got.get(timeout=10)[1]
    return th, port, err


def _req(op, n):
    return op + struct.pack("!Q", n)


def test_linktest_server_skips_probes_and_serves_the_first_valid_client():
    lt = _linktest()
    th, port, err = _start_server(lt, hello_s=0.5, accept_s=30)
    silent = socket.create_connection(("127.0.0.1", port))
    garbage = socket.create_connection(("127.0.0.1", port))
    garbage.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
    closed = socket.create_connection(("127.0.0.1", port))
    closed.close()
    huge_echo = socket.create_connection(("127.0.0.1", port))
    huge_echo.sendall(_req(b"E", lt.MAX_ECHO + 1))
    huge_bulk = socket.create_connection(("127.0.0.1", port))
    huge_bulk.sendall(_req(b"B", lt.MAX_BULK + 1))
    c = socket.create_connection(("127.0.0.1", port), timeout=20)
    c.sendall(_req(b"E", 5) + b"hello")
    assert lt.recvall(c, 5) == b"hello"
    c.sendall(_req(b"B", 1 << 20) + b"\0" * (1 << 20))
    assert lt.recvall(c, 1) == b"k"
    c.sendall(_req(b"Q", 0))
    th.join(10)
    assert not th.is_alive() and not err, err
    for s in (silent, garbage, huge_echo, huge_bulk):
        s.settimeout(5)
        try:
            assert s.recv(1) == b""                  # dropped: closed by the server
        except ConnectionResetError:
            pass
        s.close()
    c.close()


def test_linktest_server_stops_at_an_invalid_request():
    lt = _linktest()
    th, port, err = _start_server(lt, hello_s=5, accept_s=30)
    c = socket.create_connection(("127.0.0.1", port), timeout=20)
    c.sendall(_req(b"E", 3) + b"abc")
    assert lt.recvall(c, 3) == b"abc"
    c.sendall(_req(b"E", lt.MAX_ECHO + 1))          # would make the server buffer more than the cap
    th.join(10)
    assert not th.is_alive() and not err, err
    c.close()


def test_linktest_client_against_server_prints_the_same_report():
    lt = _linktest()
    th, port, err = _start_server(lt, hello_s=5, accept_s=30)
    out = io.StringIO()
    with redirect_stdout(out):
        lt.client("127.0.0.1", port, bulk=(1 << 20,), sizes=(64, 16384), rounds=20)
    th.join(10)
    assert not th.is_alive() and not err, err
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("bulk 0 GiB: ") and "Gbit/s" in lines[0]
    assert lines[1].startswith("round trip     64 B: p50 ") and " p90 " in lines[1] and " p99 " in lines[1]
    assert lines[2].startswith("round trip  16384 B: p50 ")


def test_linktest_percentiles_match_the_old_indexes():
    lt = _linktest()
    lat = list(range(2000))
    assert (lt.pct(lat, 50), lt.pct(lat, 90), lt.pct(lat, 99)) == (1000, 1800, 1980)
    assert lt.pct([7], 99) == 7


def test_linktest_usage():
    lt = _linktest()
    with redirect_stdout(io.StringIO()):
        assert lt.main([]) == 2
        assert lt.main(["client"]) == 2
        assert lt.main(["server", "1", "2", "3"]) == 2


# ---- PowerShell scripts ------------------------------------------------------------------------------------------------

needs_ps = pytest.mark.skipif(POWERSHELL is None, reason="needs Windows PowerShell")


def _ps(tmp_path, script: Path, body: str) -> str:
    """Load the function definitions of `script` (its body is never run) and run `body`; return stdout."""
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$tok = $null; $err = $null\n"
        f"$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$tok, [ref]$err)\n"
        "if ($err.Count) { throw ($err | ForEach-Object { $_.Message }) -join '; ' }\n"
        "foreach ($f in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {\n"
        "  . ([scriptblock]::Create($f.Extent.Text))\n"
        "}\n" + body + "\n", encoding="utf-8")
    done = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@needs_ps
@pytest.mark.parametrize("name", PS1)
def test_powershell_scripts_parse(tmp_path, name):
    _ps(tmp_path, _need(ROOT / name), "'parsed'")


@needs_ps
def test_setup_helpers(tmp_path):
    out = _ps(tmp_path, _need(ROOT / "setup_usb4_link.ps1"), r"""
$r = [ordered]@{}
$r.ports = ConvertTo-PortList @('5301,5302', '50052', '5301')
$r.noports = (ConvertTo-PortList @()).Count
try { ConvertTo-PortList @('70000') | Out-Null; $r.badport = 'accepted' } catch { $r.badport = 'refused' }
$r.ipok = (Test-IPv4 '10.77.0.1'), (Test-IPv4 '10.77.0.256'), (Test-IPv4 '10.77.0'), (Test-IPv4 'x')
$r.same = (Test-SameSubnet '10.77.0.1' '10.77.0.2' 24), (Test-SameSubnet '10.77.0.2' '10.78.0.1' 24), (Test-SameSubnet '10.77.0.1' '10.77.1.9' 16)
$free = [pscustomobject]@{ Name = 'free'; Static = @() }
$free2 = [pscustomobject]@{ Name = 'free2'; Static = @() }
$held = [pscustomobject]@{ Name = 'held'; Static = @('10.77.0.2') }
$pick = { param($c, $w) $x = Select-LinkAdapter $c $w; if ($x) { $x.Name } else { 'none' } }
$r.pick = (& $pick @($free, $held) '10.78.0.1'), (& $pick @($free, $held) '10.77.0.2'), (& $pick @($free, $free2) '10.78.0.1'),
          (& $pick @($free) '10.77.0.1'), (& $pick @() '10.77.0.1'), (& $pick @($held) '10.78.0.1')
$r | ConvertTo-Json -Compress
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["ports"] == [5301, 5302, 50052]          # '5301,5302' from powershell -File, de-duplicated
    assert r["noports"] == 0
    assert r["badport"] == "refused"
    assert r["ipok"] == [True, False, False, False]
    assert r["same"] == [True, False, True]
    # one adapter up: that one; held address wins; else the only adapter without a static address; else refuse
    assert r["pick"] == ["free", "held", "none", "free", "none", "held"]


@needs_ps
def test_relay_finds_the_wsl_address_in_command_output(tmp_path):
    out = _ps(tmp_path, _need(ROOT / "relay_user.ps1"), r"""
(Get-FirstIPv4 '2: eth0    inet 172.20.1.5/20 brd 172.20.15.255 scope global eth0\       valid_lft forever'),
(Get-FirstIPv4 '127.0.1.1 172.21.3.4 172.17.0.1 '), ('[' + (Get-FirstIPv4 'no address here') + ']') -join '|'
""")
    assert out.strip().splitlines()[-1] == "172.20.1.5|172.21.3.4|[]"


def _free_port() -> int:
    for _ in range(50):
        a = socket.socket(); a.bind(("127.0.0.1", 0)); port = a.getsockname()[1]
        b = socket.socket()
        try:
            b.bind(("127.0.0.2", port))
            return port
        except OSError:
            continue
        finally:
            a.close(); b.close()
    pytest.skip("no port free on both 127.0.0.1 and 127.0.0.2")


@needs_ps
def test_relay_forwards_and_closes_the_client_when_the_target_is_down():
    relay = _need(ROOT / "relay_user.ps1")
    down_port = _free_port()
    up_port = _free_port()
    while up_port == down_port:
        up_port = _free_port()
    echo = socket.socket(); echo.bind(("127.0.0.2", up_port)); echo.listen(1)

    def serve_echo():
        try:
            c, _ = echo.accept()
            with c:
                while (b := c.recv(65536)):
                    c.sendall(b)
        except OSError:                              # the relay was killed, or the test closed the socket
            pass

    threading.Thread(target=serve_echo, daemon=True).start()
    # one string '<p1>,<p2>' as powershell -File passes it; 127.0.0.2 stands in for WSL's address
    proc = subprocess.Popen([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(relay),
                             "-ListenAddress", "127.0.0.1", "-Ports", f"{down_port},{up_port}", "-WslIp", "127.0.0.2",
                             "-BridgePort", "0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout], daemon=True).start()
    log = []
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                line = lines.get(timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty:
                pytest.fail("relay did not start: " + "".join(log))
            log.append(line)
            if line.startswith("relay up:"):
                break
        assert f"127.0.0.1:{down_port},{up_port} -> WSL 127.0.0.2" in line
        # nothing listens behind down_port: the relay must close the accepted client, not leave it hanging
        c = socket.create_connection(("127.0.0.1", down_port), timeout=5)
        c.settimeout(20)
        t0 = time.monotonic()
        try:
            assert c.recv(1) == b""
        except ConnectionResetError:
            pass
        except socket.timeout:
            pytest.fail("relay left the client connected to nobody")
        assert time.monotonic() - t0 < 20
        c.close()
        # a live target behind up_port: bytes go through both ways
        c = socket.create_connection(("127.0.0.1", up_port), timeout=20)
        c.sendall(b"through the relay")
        got = b""
        while len(got) < 17:
            got += c.recv(64)
        assert got == b"through the relay"
        c.close()
    finally:
        proc.kill()
        proc.wait(timeout=30)
        echo.close()


# setup / undo run against stand-ins for every network cmdlet they call (functions win over cmdlets and netsh.exe), so
# the bodies run without touching this machine; the test refuses to run elevated, where a missed stand-in could.
_MOCKS = r"""
$ErrorActionPreference = 'Stop'
function Reset-World {
  $global:calls = New-Object System.Collections.ArrayList
  $global:adapters = @(); $global:ips = @(); $global:proxies = @(); $global:profileFails = 0
  $global:rules = New-Object System.Collections.ArrayList
}
function Log([string]$s) { [void]$global:calls.Add($s) }
function Adapter($i, $name, $desc, $status = 'Up') {
  [pscustomobject]@{ ifIndex = $i; Name = $name; InterfaceDescription = $desc; Status = $status; LinkSpeed = '20 Gbps' } }
function Ip($i, $ip, $len, $origin) {
  [pscustomobject]@{ InterfaceIndex = $i; InterfaceAlias = "if$i"; IPAddress = $ip; PrefixLength = $len; PrefixOrigin = $origin } }
function Rule($name, $group, $local = '') { [void]$global:rules.Add([pscustomobject]@{ DisplayName = $name; Group = $group; LocalAddress = $local }) }
function Get-NetAdapter { [CmdletBinding()] param([int]$InterfaceIndex, [string]$Name)
  $r = @($global:adapters)
  if ($InterfaceIndex) { $r = @($r | Where-Object { $_.ifIndex -eq $InterfaceIndex }); if (-not $r) { throw "no adapter $InterfaceIndex" } }
  if ($Name) { $r = @($r | Where-Object { $_.Name -eq $Name }); if (-not $r) { throw "no adapter $Name" } }
  $r }
function Get-NetIPAddress { [CmdletBinding()] param([int]$InterfaceIndex, [string]$AddressFamily, [string]$IPAddress)
  $r = @($global:ips)
  if ($InterfaceIndex) { $r = @($r | Where-Object { $_.InterfaceIndex -eq $InterfaceIndex }) }
  if ($IPAddress) { $r = @($r | Where-Object { $_.IPAddress -eq $IPAddress }) }
  $r }
function Remove-NetIPAddress { [CmdletBinding()] param([Parameter(ValueFromPipeline)]$InputObject, [switch]$Confirm)
  process { Log "Remove-NetIPAddress $($InputObject.IPAddress)"; $global:ips = @($global:ips | Where-Object { $_ -ne $InputObject }) } }
function New-NetIPAddress { [CmdletBinding()] param([int]$InterfaceIndex, [string]$IPAddress, [int]$PrefixLength)
  Log "New-NetIPAddress $InterfaceIndex $IPAddress/$PrefixLength"
  $global:ips += Ip $InterfaceIndex $IPAddress $PrefixLength 'Manual'; [pscustomobject]@{} }
function Set-NetIPInterface { [CmdletBinding()] param([int]$InterfaceIndex, [string]$Dhcp) Log "Set-NetIPInterface $InterfaceIndex Dhcp=$Dhcp" }
function Set-NetConnectionProfile { [CmdletBinding()] param([int]$InterfaceIndex, [string]$NetworkCategory)
  if ($global:profileFails -gt 0) { $global:profileFails--; throw 'no profile yet' }
  Log "Set-NetConnectionProfile $InterfaceIndex $NetworkCategory" }
function Get-NetFirewallRule { [CmdletBinding()] param([string]$DisplayName, [string]$Group)
  $r = @($global:rules)
  if ($DisplayName) { $r = @($r | Where-Object { $_.DisplayName -like $DisplayName }) }
  if ($Group) { $r = @($r | Where-Object { $_.Group -eq $Group }) }
  $r }
function Remove-NetFirewallRule { [CmdletBinding()] param([Parameter(ValueFromPipeline)]$InputObject)
  process { Log "Remove-NetFirewallRule $($InputObject.DisplayName)"; $global:rules.Remove($InputObject) } }
function New-NetFirewallRule { [CmdletBinding()] param([string]$DisplayName, [string]$Group, [string]$Direction, [string]$Protocol,
    $IcmpType, $LocalPort, [string]$LocalAddress, [string]$RemoteAddress, [string]$Action)
  Log "New-NetFirewallRule '$DisplayName' group=$Group $Direction $Protocol port=$LocalPort local=$LocalAddress remote=$RemoteAddress $Action"
  Rule $DisplayName $Group $LocalAddress; [pscustomobject]@{} }
function Get-NetFirewallAddressFilter { [CmdletBinding()] param([Parameter(ValueFromPipeline)]$InputObject)
  process { [pscustomobject]@{ LocalAddress = $InputObject.LocalAddress } } }
function netsh { $global:LASTEXITCODE = 0; if ($args -contains 'show') { $global:proxies } else { Log ('netsh ' + ($args -join ' ')) } }
function Test-Connection { [CmdletBinding()] param($ComputerName, [int]$Count, [switch]$Quiet) Log "Test-Connection $ComputerName"; $true }
function Start-Sleep { [CmdletBinding()] param([int]$Seconds, [int]$Milliseconds) }
function Run($File, [hashtable]$Params) {
  $text = [IO.File]::ReadAllText($File) -replace '(?m)^#Requires.*$', ''
  $sb = [scriptblock]::Create($text)
  try { $out = & $sb @Params | Out-String; $res = 'ok' } catch { $out = ''; $res = 'throw: ' + $_.Exception.Message }
  [pscustomobject]@{ result = $res; calls = @($global:calls); out = $out }
}
"""


def _scenarios(tmp_path, body: str) -> dict:
    if subprocess.run([POWERSHELL, "-NoProfile", "-Command",
                       "([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent())"
                       ".IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)"],
                      capture_output=True, text=True, timeout=60).stdout.strip() != "False":
        pytest.skip("refusing to run the setup scripts' bodies in an elevated shell")
    harness = tmp_path / "scenarios.ps1"
    harness.write_text(_MOCKS + body + "\n$res | ConvertTo-Json -Depth 5 -Compress\n", encoding="utf-8")
    done = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@needs_ps
def test_setup_usb4_link_scenarios(tmp_path):
    setup = _need(ROOT / "setup_usb4_link.ps1")
    r = _scenarios(tmp_path, f"""
$f = '{setup}'
$res = [ordered]@{{}}
Reset-World   # two nodes, node 1: one free USB4 adapter, Wi-Fi, an earlier version's rule and a stale port rule
$global:adapters = @((Adapter 7 'Ethernet 3' 'USB4(TM) P2P Network Adapter'), (Adapter 3 'Wi-Fi' 'Wireless'))
$global:ips = @((Ip 7 '169.254.3.4' 16 'WellKnown'), (Ip 3 '192.168.1.5' 24 'Dhcp'), (Ip 1 '127.0.0.1' 8 'WellKnown'))
Rule 'N1X cluster ping' ''; Rule 'hive-ram 10.77.0.1 tcp 9999' 'hive-ram' '10.77.0.1'; Rule 'hive-ram 10.78.0.1 icmp' 'hive-ram' '10.78.0.1'
$global:profileFails = 2
$res.node1 = Run $f @{{ Node = '1' }}
Reset-World   # two USB4 adapters up, both free: refuse to guess
$global:adapters = @((Adapter 7 'Ethernet 3' 'USB4(TM) P2P Network Adapter'), (Adapter 8 'Ethernet 4' 'USB4(TM) P2P Network Adapter #2'))
$res.ambiguous = Run $f @{{ Address = '10.78.0.1'; PeerAddress = '10.78.0.2' }}
Reset-World   # node A's second cable: the first adapter already holds 10.77.0.2; ports as powershell -File passes them
$global:adapters = @((Adapter 7 'Ethernet 3' 'USB4(TM) P2P Network Adapter'), (Adapter 8 'Ethernet 4' 'USB4(TM) P2P Network Adapter #2'))
$global:ips = @((Ip 7 '10.77.0.2' 24 'Manual'), (Ip 8 '169.254.9.9' 16 'WellKnown'))
Rule 'N1X cluster ping' ''
$res.second = Run $f @{{ Address = '10.78.0.1'; PeerAddress = '10.78.0.2'; AllowTcpPorts = @('5302,5303') }}
Reset-World   # the same subnet on two adapters
$global:adapters = @((Adapter 7 'Ethernet 3' 'USB4(TM) P2P Network Adapter'), (Adapter 8 'Ethernet 4' 'USB4(TM) P2P Network Adapter #2'))
$global:ips = @((Ip 7 '10.77.0.2' 24 'Manual'))
$res.overlap = Run $f @{{ Address = '10.77.0.5'; PeerAddress = '10.77.0.6'; InterfaceIndex = 8 }}
Reset-World   # the older portproxy setup, opt-in
$global:adapters = @((Adapter 7 'Ethernet 3' 'USB4(TM) P2P Network Adapter'))
$res.publish = Run $f @{{ Node = '1'; PublishPorts = @('50052') }}
Reset-World   # no cable
$global:adapters = @((Adapter 7 'Ethernet 3' 'USB4(TM) P2P Network Adapter' 'Disconnected'))
$res.nocable = Run $f @{{ Node = '2' }}
Reset-World   # peers in different subnets
$res.badpeer = Run $f @{{ Address = '10.77.0.1'; PeerAddress = '10.78.0.2' }}
""")
    n1 = r["node1"]
    assert n1["result"] == "ok", n1
    assert n1["calls"] == [
        "Set-NetIPInterface 7 Dhcp=Disabled",
        "Remove-NetIPAddress 169.254.3.4",
        "New-NetIPAddress 7 10.77.0.1/24",
        "Set-NetConnectionProfile 7 Private",                       # after two not-ready attempts
        "Remove-NetFirewallRule hive-ram 10.77.0.1 tcp 9999",       # this address's old rules only
        "Remove-NetFirewallRule N1X cluster ping",                  # an earlier version's rule (10.77.0.x link)
        "New-NetFirewallRule 'hive-ram 10.77.0.1 icmp' group=hive-ram Inbound ICMPv4 port= local=10.77.0.1 remote=10.77.0.2 Allow",
        "New-NetFirewallRule 'hive-ram 10.77.0.1 tcp 50052' group=hive-ram Inbound TCP port=50052 local=10.77.0.1 remote=10.77.0.2 Allow",
        "New-NetFirewallRule 'hive-ram 10.77.0.1 tcp 5201' group=hive-ram Inbound TCP port=5201 local=10.77.0.1 remote=10.77.0.2 Allow",
        "Test-Connection 10.77.0.2",
    ]
    assert "peer 10.77.0.2 answers ping" in n1["out"]
    amb = r["ambiguous"]
    assert amb["result"].startswith("throw: cannot tell which USB4 adapter") and "-InterfaceIndex 8" in amb["result"]
    assert amb["calls"] == []
    sec = r["second"]
    assert sec["result"] == "ok", sec
    assert "New-NetIPAddress 8 10.78.0.1/24" in sec["calls"] and "Set-NetIPInterface 8 Dhcp=Disabled" in sec["calls"]
    assert not any(c.startswith(("Set-NetIPInterface 7", "Remove-NetIPAddress 10.77")) for c in sec["calls"])
    assert [c for c in sec["calls"] if "New-NetFirewallRule" in c] == [
        "New-NetFirewallRule 'hive-ram 10.78.0.1 icmp' group=hive-ram Inbound ICMPv4 port= local=10.78.0.1 remote=10.78.0.2 Allow",
        "New-NetFirewallRule 'hive-ram 10.78.0.1 tcp 5302' group=hive-ram Inbound TCP port=5302 local=10.78.0.1 remote=10.78.0.2 Allow",
        "New-NetFirewallRule 'hive-ram 10.78.0.1 tcp 5303' group=hive-ram Inbound TCP port=5303 local=10.78.0.1 remote=10.78.0.2 Allow",
    ]
    assert "Remove-NetFirewallRule N1X cluster ping" not in sec["calls"]    # the 10.77 link's rule stays
    ov = r["overlap"]
    assert ov["result"].startswith("throw: 10.77.0.5/24 overlaps 10.77.0.2/24") and ov["calls"] == []
    pub = r["publish"]
    assert pub["result"] == "ok", pub
    assert "netsh interface portproxy delete v4tov4 listenaddress=10.77.0.1 listenport=50052" in pub["calls"]
    assert "netsh interface portproxy add v4tov4 listenaddress=10.77.0.1 listenport=50052 connectaddress=127.0.0.1 connectport=50052" in pub["calls"]
    assert sum("tcp 50052" in c for c in pub["calls"]) == 1
    assert r["nocable"]["result"].startswith("throw: no USB4 network adapter is up") and r["nocable"]["calls"] == []
    assert r["badpeer"]["result"] == "throw: 10.77.0.1 and 10.78.0.2 are not in one /24 subnet"
    assert not any(c.startswith("netsh") for c in n1["calls"] + sec["calls"])    # portproxy only when asked


@needs_ps
def test_undo_usb4_link_scenarios(tmp_path):
    undo = _need(ROOT / "undo_usb4_link.ps1")
    r = _scenarios(tmp_path, f"""
$f = '{undo}'
$res = [ordered]@{{}}
function World {{
  Reset-World
  $global:adapters = @((Adapter 7 'Ethernet 3' 'USB4(TM) P2P Network Adapter'), (Adapter 8 'Ethernet 4' 'USB4(TM) P2P Network Adapter #2'))
  Rule 'hive-ram 10.77.0.2 icmp' 'hive-ram' '10.77.0.2'; Rule 'hive-ram 10.77.0.2 tcp 5301' 'hive-ram' '10.77.0.2'
  Rule 'tfpp-glm-5301' '' '10.77.0.2'; Rule 'tfpp-glm-5302' '' '10.78.0.2'; Rule 'N1X cluster ping' ''
  $global:proxies = @('Listen on ipv4:             Connect to ipv4:', '', 'Address         Port        Address         Port',
    '--------------- ----------  --------------- ----------', '10.77.0.2       5201        127.0.0.1       5201',
    '0.0.0.0         6000        172.20.1.5      6000')
}}
World         # one link
$global:ips = @((Ip 7 '10.77.0.2' 24 'Manual'))
$res.one = Run $f @{{}}
World         # two links: name one, or -All
$global:ips = @((Ip 7 '10.77.0.2' 24 'Manual'), (Ip 8 '10.78.0.1' 24 'Manual'))
$res.two = Run $f @{{}}
$res.named = Run $f @{{ Address = '10.78.0.1' }}
World
$global:ips = @((Ip 7 '10.77.0.2' 24 'Manual'), (Ip 8 '10.78.0.1' 24 'Manual'))
$res.all = Run $f @{{ All = $true }}
""")
    one = r["one"]
    assert one["result"] == "ok", one
    assert one["calls"] == [
        "Remove-NetFirewallRule hive-ram 10.77.0.2 icmp",
        "Remove-NetFirewallRule hive-ram 10.77.0.2 tcp 5301",
        "Remove-NetFirewallRule tfpp-glm-5301",                     # the relay rule scoped to this address only
        "netsh interface portproxy delete v4tov4 listenaddress=10.77.0.2 listenport=5201",
        "Remove-NetIPAddress 10.77.0.2",
        "Set-NetConnectionProfile 7 Public",
        "Set-NetIPInterface 7 Dhcp=Enabled",
        "Remove-NetFirewallRule N1X cluster ping",
    ]
    assert "link settings removed" in one["out"] and "FAILED" not in one["out"]
    assert r["two"]["result"].startswith("throw: several links on this node (10.77.0.2, 10.78.0.1)") and r["two"]["calls"] == []
    named = r["named"]
    assert named["result"] == "ok", named
    assert named["calls"] == ["Remove-NetIPAddress 10.78.0.1", "Set-NetConnectionProfile 8 Public", "Set-NetIPInterface 8 Dhcp=Enabled"]
    every = r["all"]
    assert every["result"] == "ok", every
    assert "Remove-NetIPAddress 10.77.0.2" in every["calls"] and "Remove-NetIPAddress 10.78.0.1" in every["calls"]
    assert "Remove-NetFirewallRule N1X cluster ping" in every["calls"]
    assert "Remove-NetFirewallRule tfpp-glm-5302" not in every["calls"]      # scoped to an address not set up here


# ---- scripts/flashnext/env.sh -------------------------------------------------------------------------------------------

def _bash():
    cands = []
    if os.name == "nt":
        for base in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles")):
            if base:
                cands.append(os.path.join(base, "Git", "bin", "bash.exe"))
    found = shutil.which("bash")
    if found:
        cands.append(found)
    for c in cands:
        if os.path.isfile(c):
            return c
    pytest.skip("needs bash")


def _run_bash(script: str) -> subprocess.CompletedProcess:
    # the script goes in on stdin: no Windows path ever reaches bash (a WSL bash would mangle it)
    return subprocess.run([_bash(), "-s"], input=script.encode(), capture_output=True, timeout=120)


def test_env_sh_is_lf_and_parses():
    data = _need(ENV_SH).read_bytes()
    assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")
    done = subprocess.run([_bash(), "-n"], input=data, capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr.decode(errors="replace")


def test_env_sh_points_at_its_own_checkout_and_the_venv():
    text = _need(ENV_SH).read_text(encoding="utf-8")
    assert "ENV_SH_EOF" not in text
    done = _run_bash(f"""
set -u
T=$(mktemp -d)
mkdir -p "$T/repo/tfpp/scripts/flashnext" "$T/venv/bin" "$T/venv/lib/python3.14/site-packages/nvidia/cu13"
cat > "$T/repo/tfpp/scripts/flashnext/env.sh" <<'ENV_SH_EOF'
{text}ENV_SH_EOF
printf '#!/bin/sh\\necho "%s"\\n' "$T/venv/lib/python3.14/site-packages" > "$T/venv/bin/python"
chmod +x "$T/venv/bin/python"
unset TFPP_DIR PYTHONPATH CUDA_HOME
export TFPP_QWEN_VENV=$T/none
. "$T/repo/tfpp/scripts/flashnext/env.sh" 2>/dev/null
echo "missing_rc=$?"
export TFPP_QWEN_VENV=$T/venv
. "$T/repo/tfpp/scripts/flashnext/env.sh"
echo "sourced_rc=$?"
[[ $PYTHONPATH == "$T/repo/tfpp" ]] && echo PYTHONPATH_OK
[[ $CUDA_HOME == "$T/venv/lib/python3.14/site-packages/nvidia/cu13" ]] && echo CUDA_HOME_OK
[[ $CPLUS_INCLUDE_PATH == *"/include/python3.14"* ]] && echo PYINC_OK
[[ $TORCH_CUDA_ARCH_LIST == 12.1 && $HF_HUB_OFFLINE == 1 ]] && echo KNOBS_OK
[[ -z ${{_tfpp_dir+x}}${{_tfpp_v+x}}${{TFPP_DIR+x}} ]] && echo NO_LEAKS
rm -rf "$T"
""")
    out = done.stdout.decode(errors="replace")
    assert done.returncode == 0, out + done.stderr.decode(errors="replace")
    for want in ("missing_rc=1", "sourced_rc=0", "PYTHONPATH_OK", "CUDA_HOME_OK", "PYINC_OK", "KNOBS_OK", "NO_LEAKS"):
        assert want in out, out + done.stderr.decode(errors="replace")
