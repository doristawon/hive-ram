# hive-ram: TCP relay on the Windows side of a node whose WSL runs in NAT mode. It listens on this node's USB4 link
# address and forwards each connection to the same port on WSL's own address. No admin and no portproxy needed; keep
# the session open (closing it or Ctrl+C stops the relay) and start it again after WSL restarts (WSL's address changes).
#   powershell -ExecutionPolicy Bypass -File relay_user.ps1                    # 10.77.0.1, ports 50052 (rpc) and 5201
#   powershell -ExecutionPolicy Bypass -File relay_user.ps1 -ListenAddress 10.77.0.2 -Ports 5301
# Measured on the hardware: Windows portproxy (iphlpsvc) 0.42 Gbit/s, this relay 10.1 Gbit/s on one node, and
# WSL -> USB4 -> relay -> WSL across two nodes 5.2 Gbit/s, 0.49 ms p50 round trip for 64 B (linktest.py).
# A netsh portproxy entry on the same address and port holds the port and the bind fails: remove it first
# (start_n1x_relay.ps1 does that, as admin).
# WSL's address: -WslIp if given; else -BridgePort N reads it from an existing portproxy entry 0.0.0.0:N -> <ip>:<port>
# (off by default; useful under a Windows account without its own WSL distro); else asked from WSL itself (eth0, then
# hostname -I; -Distro picks the distro).
# Only for WSL's default NAT networking: with mirrored networking WSL already holds the link address.
# Exposure: the firewall rule from setup_usb4_link.ps1 admits only the link peer to this relay, but the server inside
# WSL that it forwards to listens on WSL's address, which every process on this host and every WSL distro can reach.
# llama.cpp's rpc-server and the tfpp links have no authentication: run them only while testing, on a host you trust.
param(
  [string]$ListenAddress = '10.77.0.1',
  [Alias('Port')][string[]]$Ports = @('50052', '5201'),
  [string]$WslIp = '',
  [int]$BridgePort = 0,
  [string]$Distro = ''
)
$ErrorActionPreference = 'Stop'

# '5301,5302' arrives as one string through powershell -File; accept lists, commas and spaces alike.
function ConvertTo-PortList([string[]]$Items) {
  $out = New-Object System.Collections.Generic.List[int]
  foreach ($s in $Items) {
    foreach ($p in ([string]$s -split '[,\s]+')) {
      if (-not $p) { continue }
      $n = 0
      if (-not [int]::TryParse($p, [ref]$n) -or $n -lt 1 -or $n -gt 65535) { throw "not a TCP port: '$p'" }
      if (-not $out.Contains($n)) { $out.Add($n) }
    }
  }
  if ($out.Count -eq 0) { throw 'no ports given' }
  return , $out.ToArray()
}

# the first IPv4 address in text such as 'ip -4 -o addr' or 'hostname -I' output, loopback excluded
function Get-FirstIPv4([string]$Text) {
  foreach ($t in ($Text -split '[\s/]+')) {
    if ($t -match '^\d{1,3}(\.\d{1,3}){3}$' -and $t -notlike '127.*') { return $t }
  }
  return ''
}

function Resolve-WslIp([string]$Given, [int]$Bridge, [string]$Name) {
  if ($Given) { return $Given }
  if ($Bridge -gt 0) {
    $m = (netsh interface portproxy show v4tov4) | Select-String ('^\s*0\.0\.0\.0\s+' + $Bridge + '\s+(\d+\.\d+\.\d+\.\d+)\s+22\s*$')
    if ($m) { return $m.Matches[0].Groups[1].Value }
  }
  $pick = @(); if ($Name) { $pick = @('-d', $Name) }
  foreach ($cmd in @(, @('ip', '-4', '-o', 'addr', 'show', 'dev', 'eth0')) + @(, @('hostname', '-I'))) {
    try { $out = & wsl.exe @pick -e @cmd 2>$null } catch { $out = $null }
    $ip = Get-FirstIPv4 ([string]($out -join ' '))
    if ($ip) { return $ip }
  }
  return ''
}

$portList = ConvertTo-PortList $Ports
$WslIp = Resolve-WslIp $WslIp $BridgePort $Distro
if (-not $WslIp) { throw "no WSL address found: start WSL first, or pass -WslIp (inside WSL: ip -4 -o addr show eth0)" }
$listeners = foreach ($p in $portList) { $l = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse($ListenAddress), $p); $l.Start(); $l }
$accept = @{}; foreach ($l in $listeners) { $accept[$l.LocalEndpoint.Port] = $l.AcceptTcpClientAsync() }
$conns = New-Object System.Collections.ArrayList
"relay up: ${ListenAddress}:$($portList -join ',') -> WSL $WslIp"
try {
  while ($true) {
    foreach ($l in $listeners) {
      $port = $l.LocalEndpoint.Port; $t = $accept[$port]
      if ($t.IsCompleted) {
        $a = $null; $b = $null
        try {
          $a = $t.Result
          $b = [Net.Sockets.TcpClient]::new($WslIp, $port)
          foreach ($c in $a, $b) { $c.NoDelay = $true; $c.ReceiveBufferSize = 4MB; $c.SendBufferSize = 4MB }
          $sa = $a.GetStream(); $sb = $b.GetStream()
          [void]$conns.Add([pscustomobject]@{ Port = $port; A = $a; B = $b; Up = $sa.CopyToAsync($sb, 1MB); Down = $sb.CopyToAsync($sa, 1MB); Since = Get-Date })
          "{0:HH:mm:ss} open {1} -> :{2}" -f (Get-Date), $a.Client.RemoteEndPoint, $port
        } catch {
          $why = $_.Exception.Message
          # close what was opened: a half-made pair would leave the client connected to nobody, waiting forever
          foreach ($x in $a, $b) { if ($x) { try { $x.Close() } catch {} } }
          "{0:HH:mm:ss} forward failed :{1} {2} (client closed)" -f (Get-Date), $port, $why
        }
        $accept[$port] = $l.AcceptTcpClientAsync()
      }
    }
    foreach ($c in @($conns)) {
      if ($c.Up.IsCompleted -or $c.Down.IsCompleted) {
        foreach ($x in $c.A, $c.B) { try { $x.Close() } catch {} }
        [void]$conns.Remove($c)
        "{0:HH:mm:ss} closed :{1} ({2:N0} s)" -f (Get-Date), $c.Port, ((Get-Date) - $c.Since).TotalSeconds
      }
    }
    Start-Sleep -Milliseconds 5
  }
} finally {
  foreach ($l in $listeners) { $l.Stop() }
  foreach ($c in $conns) { foreach ($x in $c.A, $c.B) { try { $x.Close() } catch {} } }
}
