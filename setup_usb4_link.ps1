#Requires -RunAsAdministrator
# hive-ram: one point-to-point USB4 / Thunderbolt link, Windows side. Run in an ADMIN PowerShell on each end after the
# cable is plugged straight between the two machines' USB4 ports (not through a dock).
# Two nodes, one cable (the setup run on hardware):
#   node 1:  powershell -ExecutionPolicy Bypass -File setup_usb4_link.ps1 -Node 1   # 10.77.0.1, runs rpc-server in WSL
#   node 2:  powershell -ExecutionPolicy Bypass -File setup_usb4_link.ps1 -Node 2   # 10.77.0.2, runs llama-server
# Any link: one subnet per cable, and a node with two cables runs this once per adapter. The three-node tfpp-glm chain
# C -> A -> B (written, not yet run on hardware), with the stage ports each receiving node's relay listens on:
#   C:  ... -Address 10.77.0.1 -PeerAddress 10.77.0.2
#   A:  ... -Address 10.77.0.2 -PeerAddress 10.77.0.1 -AllowTcpPorts 5301 -InterfaceAlias '<A port cabled to C>'
#   A:  ... -Address 10.78.0.1 -PeerAddress 10.78.0.2 -InterfaceAlias '<A port cabled to B>'
#   B:  ... -Address 10.78.0.2 -PeerAddress 10.78.0.1 -AllowTcpPorts 5302
# On the chosen adapter only: a static IPv4 address (DHCP off), network category Private, and inbound firewall rules
# (group 'hive-ram', named 'hive-ram <address> ...') that admit ICMP echo and the given TCP ports only from the peer's
# address and only to this link's address. Running it again replaces that address's rules. Undo: undo_usb4_link.ps1.
# Nothing here touches Wi-Fi, VPN or other virtual adapters. With several USB4 adapters up and none named
# (-InterfaceAlias / -InterfaceIndex), it takes the one already holding the address, else the only one without a
# static address, and otherwise refuses to guess.
# -PublishPorts (opt-in; the first setup did this on node 1): netsh portproxy from <Address>:<port> to
# <PublishTo>:<port> (default 127.0.0.1, which needs WSL's localhost forwarding). relay_user.ps1 replaces it: portproxy
# measured 0.42 Gbit/s against 10.1 Gbit/s for the relay, and the relay cannot bind a port a portproxy entry holds.
# Exposure: these rules guard the link side only. Windows 127.0.0.1:<port>, and the WSL address a relay or portproxy
# forwards to, are reachable by every process on this host and by every WSL distro, whatever the firewall says.
# llama.cpp's rpc-server has no authentication or encryption (a client can read and write its memory buffers): start
# it only while benchmarking, on a host you trust, and stop it afterwards.
[CmdletBinding(DefaultParameterSetName = 'Preset')]
param(
  [Parameter(ParameterSetName = 'Preset', Mandatory)][ValidateSet('1', '2')][string]$Node,
  [Parameter(ParameterSetName = 'Explicit', Mandatory)][string]$Address,
  [Parameter(ParameterSetName = 'Explicit', Mandatory)][string]$PeerAddress,
  [ValidateRange(8, 30)][int]$PrefixLength = 24,
  [string]$InterfaceAlias = '',
  [int]$InterfaceIndex = 0,
  [string[]]$AllowTcpPorts = @(),
  [string[]]$PublishPorts = @(),
  [string]$PublishTo = '127.0.0.1'
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
  return , $out.ToArray()
}

function Test-IPv4([string]$Ip) {
  if ($Ip -notmatch '^\d{1,3}(\.\d{1,3}){3}$') { return $false }
  foreach ($o in $Ip.Split('.')) { if ([int]$o -gt 255) { return $false } }
  return $true
}

# the network number of an IPv4 address under a prefix length, as an integer
function Get-IPv4Network([string]$Ip, [int]$Length) {
  if (-not (Test-IPv4 $Ip)) { throw "not an IPv4 address: '$Ip'" }
  $o = $Ip.Split('.') | ForEach-Object { [long]$_ }
  $n = $o[0] * 16777216 + $o[1] * 65536 + $o[2] * 256 + $o[3]
  $mask = [long]4294967296 - [long][math]::Pow(2, 32 - $Length)
  return ($n -band $mask)
}

function Test-SameSubnet([string]$A, [string]$B, [int]$Length) {
  return (Get-IPv4Network $A $Length) -eq (Get-IPv4Network $B $Length)
}

# $Candidates: the USB4 adapters that are up, each with Static = the manually set IPv4 addresses it holds now
function Select-LinkAdapter($Candidates, [string]$Want) {
  $all = @($Candidates | Where-Object { $_ })
  if ($all.Count -eq 1) { return $all[0] }
  $mine = @($all | Where-Object { @($_.Static | Where-Object { $_ }) -contains $Want })
  if ($mine.Count -eq 1) { return $mine[0] }
  $free = @($all | Where-Object { @($_.Static | Where-Object { $_ }).Count -eq 0 })
  if ($free.Count -eq 1) { return $free[0] }
  return $null
}

$preset = $PSCmdlet.ParameterSetName -eq 'Preset'
if ($preset) {
  $Address = @{ '1' = '10.77.0.1'; '2' = '10.77.0.2' }[$Node]
  $PeerAddress = @{ '1' = '10.77.0.2'; '2' = '10.77.0.1' }[$Node]
  if (-not $PSBoundParameters.ContainsKey('AllowTcpPorts')) {
    # node 1: llama.cpp rpc-server (50052) and the link test (5201); node 2: the link test
    $AllowTcpPorts = @{ '1' = @('50052', '5201'); '2' = @('5201') }[$Node]
  }
}
foreach ($x in $Address, $PeerAddress) { if (-not (Test-IPv4 $x)) { throw "not an IPv4 address: '$x'" } }
if ($Address -eq $PeerAddress) { throw "-Address and -PeerAddress are both $Address" }
if (-not (Test-SameSubnet $Address $PeerAddress $PrefixLength)) { throw "$Address and $PeerAddress are not in one /$PrefixLength subnet" }
$tcp = ConvertTo-PortList $AllowTcpPorts
$publish = ConvertTo-PortList $PublishPorts

# ---- the adapter -------------------------------------------------------------------------------------------------
$up = @(Get-NetAdapter | Where-Object { $_.InterfaceDescription -match 'USB4|P2P' -and $_.Status -eq 'Up' })
$static = {
  param($Index)
  @(Get-NetIPAddress -InterfaceIndex $Index -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.PrefixOrigin -eq 'Manual' } | ForEach-Object { $_.IPAddress })
}
if ($InterfaceIndex -gt 0) { $a = Get-NetAdapter -InterfaceIndex $InterfaceIndex }
elseif ($InterfaceAlias) { $a = Get-NetAdapter -Name $InterfaceAlias }
else {
  if ($up.Count -eq 0) {
    throw 'no USB4 network adapter is up: plug the cable straight between both machines'' USB4 / Thunderbolt ports, with both switched on and signed in'
  }
  $cands = @(foreach ($x in $up) { [pscustomobject]@{ Adapter = $x; Static = @(& $static $x.ifIndex) } })
  $pick = Select-LinkAdapter $cands $Address
  if (-not $pick) {
    $list = ($cands | ForEach-Object {
        "  -InterfaceIndex $($_.Adapter.ifIndex)  '$($_.Adapter.Name)'  $($_.Adapter.LinkSpeed)  static: $(if (@($_.Static).Count) { @($_.Static) -join ', ' } else { 'none' })"
      }) -join "`n"
    throw "cannot tell which USB4 adapter is this link; name it with -InterfaceAlias or -InterfaceIndex:`n$list"
  }
  $a = $pick.Adapter
}
if ($a.Status -ne 'Up') { Write-Warning "adapter '$($a.Name)' is $($a.Status), not Up" }
"USB4 adapter: $($a.Name) | $($a.InterfaceDescription) | $($a.LinkSpeed) (ifIndex $($a.ifIndex))"

# one subnet per link: refuse when another interface already holds an address in this link's subnet (a node with two
# cables needs two subnets)
foreach ($ip in @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue)) {
  if ($ip.InterfaceIndex -eq $a.ifIndex -or -not (Test-IPv4 $ip.IPAddress)) { continue }
  if (Test-SameSubnet $ip.IPAddress $Address $PrefixLength) {
    throw "$Address/$PrefixLength overlaps $($ip.IPAddress)/$($ip.PrefixLength) on '$($ip.InterfaceAlias)': give this link its own subnet (e.g. 10.78.0.x)"
  }
}

Set-NetIPInterface -InterfaceIndex $a.ifIndex -Dhcp Disabled
Get-NetIPAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Remove-NetIPAddress -Confirm:$false
New-NetIPAddress -InterfaceIndex $a.ifIndex -IPAddress $Address -PrefixLength $PrefixLength | Out-Null
# Windows may not have made the connection profile yet right after the address change: retry for about 10 s
$ok = $false
for ($i = 0; $i -lt 20 -and -not $ok; $i++) {
  try { Set-NetConnectionProfile -InterfaceIndex $a.ifIndex -NetworkCategory Private; $ok = $true }
  catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ok) { Write-Warning 'could not set the network category to Private (the firewall rules below do not depend on it)' }
"address: $Address/$PrefixLength (peer $PeerAddress)"

# ---- firewall: only the peer, only to this link's address ---------------------------------------------------------
Get-NetFirewallRule -DisplayName "hive-ram $Address *" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
if ($Address -like '10.77.0.*') {
  # rules an earlier version of this script made (no group, no local address)
  Get-NetFirewallRule -DisplayName 'N1X cluster*' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
}
New-NetFirewallRule -DisplayName "hive-ram $Address icmp" -Group 'hive-ram' -Direction Inbound -Protocol ICMPv4 -IcmpType 8 `
  -LocalAddress $Address -RemoteAddress $PeerAddress -Action Allow | Out-Null
foreach ($p in @($tcp) + @($publish | Where-Object { $tcp -notcontains $_ })) {
  New-NetFirewallRule -DisplayName "hive-ram $Address tcp $p" -Group 'hive-ram' -Direction Inbound -Protocol TCP -LocalPort $p `
    -LocalAddress $Address -RemoteAddress $PeerAddress -Action Allow | Out-Null
}
"firewall: ping$(($tcp + $publish | Sort-Object -Unique | ForEach-Object { ", tcp $_" }) -join '') from $PeerAddress only"

foreach ($p in $publish) {
  netsh interface portproxy delete v4tov4 listenaddress=$Address listenport=$p | Out-Null
  netsh interface portproxy add v4tov4 listenaddress=$Address listenport=$p connectaddress=$PublishTo connectport=$p | Out-Null
  "portproxy: ${Address}:$p -> ${PublishTo}:$p"
}

Start-Sleep 3
if (Test-Connection $PeerAddress -Count 2 -Quiet) { "peer $PeerAddress answers ping" }
else {
  "peer $PeerAddress does not answer ping yet: run this script on the other end too"
  if ($up.Count -gt 1) { "(several USB4 adapters are up: if the cable is in the other port, run again with -InterfaceAlias)" }
}
