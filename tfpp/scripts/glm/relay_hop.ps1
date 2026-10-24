# tfpp-glm: one inbound hop's relay on the receiving N1X's Windows side, from its USB4 address into WSL (NAT).
# Any listen address and ports (setup_usb4_link.ps1's fixed 10.77.0.x as parameters), no admin, no portproxy. Run it
# on s1 (A) and on the leader (B); stage 0 (C) needs none. Keep the window open.
#   A:  powershell -ExecutionPolicy Bypass -File relay_hop.ps1 -ListenAddress 10.77.0.2 -Ports 5301
#   B:  powershell -ExecutionPolicy Bypass -File relay_hop.ps1 -ListenAddress 10.78.0.2 -Ports 5302
# Firewall (a one-time admin step, only the upstream peer allowed), e.g. on A:
#   New-NetFirewallRule -DisplayName tfpp-glm-5301 -Direction Inbound -Protocol TCP -LocalPort 5301 `
#       -LocalAddress 10.77.0.2 -RemoteAddress 10.77.0.1 -Action Allow
# Measured through a userland relay of this kind (relay_user.ps1): 5.2 Gbit/s, 0.49 ms p50 one way.
# Nothing listening in WSL yet: the upstream connection is closed at once (the stage before sees EOF and retries).
# Both sides keep TCP keepalive (60 s idle, 10 s probes): a peer N1X that lost power is noticed while idle, the
# pair closes, and the stage behind this relay stops the chain (fail-stop) instead of waiting for the next request.
# WSL's address: -WslIp, else 'wsl.exe -e hostname -I' as this Windows user. WSL VMs are per user: a relay started
# from another account than the distro's owner passes -WslIp, or -PortproxyPort N to read the address from an
# existing 'netsh interface portproxy' rule that listens on 0.0.0.0:N and forwards into WSL. WSL's NAT address
# changes when its VM restarts: a connect into WSL that times out or finds no route looks the address up again (at
# most every 10 s; never with -WslIp; a refused connect only means the stage is not listening yet). Otherwise restart
# this relay after a WSL restart.
param(
  [Parameter(Mandatory)][string]$ListenAddress,
  [Parameter(Mandatory)][Alias('Port')][int[]]$Ports,
  [string]$WslIp = '',
  [int]$PortproxyPort = 0,
  [int]$ConnectTimeoutMs = 5000
)
$ErrorActionPreference = 'Stop'
$pinned = [bool]$WslIp
function Resolve-WslIp {
  if ($PortproxyPort -gt 0) {
    $pattern = '^\s*0\.0\.0\.0\s+' + $PortproxyPort + '\s+(\d+\.\d+\.\d+\.\d+)\s+\d+\s*$'
    $m = (netsh interface portproxy show v4tov4) | Select-String $pattern
    if ($m) { return $m.Matches[0].Groups[1].Value }
    return ''
  }
  $ip = (wsl.exe -e hostname -I) -split '\s+' | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -First 1
  if ($ip) { return [string]$ip }
  return ''
}
if (-not $WslIp) { $WslIp = Resolve-WslIp }
if (-not $WslIp) { throw 'no WSL address: start WSL first, or pass -WslIp (or -PortproxyPort)' }
$lastLookup = Get-Date
# tcp_keepalive {u32 onoff, u32 keepalivetime ms, u32 keepaliveinterval ms} for SIO_KEEPALIVE_VALS
$keepAlive = [byte[]]::new(12)
[BitConverter]::GetBytes([uint32]1).CopyTo($keepAlive, 0)
[BitConverter]::GetBytes([uint32]60000).CopyTo($keepAlive, 4)
[BitConverter]::GetBytes([uint32]10000).CopyTo($keepAlive, 8)
function Set-Link($c) {
  $c.NoDelay = $true; $c.ReceiveBufferSize = 8MB; $c.SendBufferSize = 8MB
  try {                                  # best effort: a relay without keepalive still forwards
    $c.Client.SetSocketOption([Net.Sockets.SocketOptionLevel]::Socket, [Net.Sockets.SocketOptionName]::KeepAlive, $true)
    [void]$c.Client.IOControl([Net.Sockets.IOControlCode]::KeepAliveValues, $keepAlive, $null)
  } catch { "{0:HH:mm:ss} keepalive not set: {1}" -f (Get-Date), $_.Exception.Message }
}
$listeners = foreach ($p in $Ports) { $l = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse($ListenAddress), $p); $l.Start(); $l }
$accept = @{}; foreach ($l in $listeners) { $accept[$l.LocalEndpoint.Port] = $l.AcceptTcpClientAsync() }
$conns = New-Object System.Collections.ArrayList
"relay up: ${ListenAddress}:$($Ports -join ',') -> WSL $WslIp"
try {
  while ($true) {
    foreach ($l in $listeners) {
      $port = $l.LocalEndpoint.Port; $t = $accept[$port]
      if ($t.IsCompleted) {
        $a = $null; $b = $null; $connecting = $false
        try {
          $a = $t.Result
          # a bounded connect: a stale WSL address must not hold this loop (and every other port) for the ~21 s of
          # Windows' SYN retries; a live one answers at once, or refuses within ~2 s (Windows retries a refused SYN)
          $connecting = $true
          $b = [Net.Sockets.TcpClient]::new()
          if (-not $b.ConnectAsync($WslIp, $port).Wait($ConnectTimeoutMs)) {
            throw [TimeoutException]::new("connect to WSL ${WslIp}:$port timed out after $ConnectTimeoutMs ms")
          }
          foreach ($c in $a, $b) { Set-Link $c }
          $sa = $a.GetStream(); $sb = $b.GetStream()
          [void]$conns.Add([pscustomobject]@{ Port = $port; A = $a; B = $b; Up = $sa.CopyToAsync($sb, 1MB); Down = $sb.CopyToAsync($sa, 1MB); Since = Get-Date })
          "{0:HH:mm:ss} open {1} -> :{2}" -f (Get-Date), $a.Client.RemoteEndPoint, $port
        } catch {
          $e = $_.Exception.GetBaseException()
          $why = $e.Message
          # close what was opened: a half-made pair would leave the stage before connected to nobody
          foreach ($x in $a, $b) { if ($x) { try { $x.Close() } catch {} } }
          "{0:HH:mm:ss} forward failed :{1} {2} (upstream closed; it retries)" -f (Get-Date), $port, $why
          $refused = ($e -is [Net.Sockets.SocketException]) -and ($e.SocketErrorCode -eq [Net.Sockets.SocketError]::ConnectionRefused)
          if ($connecting -and -not $pinned -and -not $refused -and ((Get-Date) - $lastLookup).TotalSeconds -ge 10) {
            $lastLookup = Get-Date
            $new = ''
            try { $new = Resolve-WslIp } catch { "{0:HH:mm:ss} WSL address lookup failed: {1}" -f (Get-Date), $_.Exception.Message }
            if ($new -and $new -ne $WslIp) { "{0:HH:mm:ss} WSL address {1} -> {2}" -f (Get-Date), $WslIp, $new; $WslIp = $new }
          }
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
