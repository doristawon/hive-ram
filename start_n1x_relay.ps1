#Requires -RunAsAdministrator
# hive-ram: admin variant of relay_user.ps1. It first removes the netsh portproxy entries on the same address and ports
# (an older setup_usb4_link.ps1 -PublishPorts made them; they hold the ports, and portproxy measured 0.42 Gbit/s against
# 10.1 Gbit/s for the relay), then runs relay_user.ps1 with the same arguments. Once those entries are gone,
# relay_user.ps1 alone is enough and needs no admin. Keep the window open while testing (Ctrl+C or closing it stops it).
#   powershell -ExecutionPolicy Bypass -File .\start_n1x_relay.ps1             # 10.77.0.1, ports 50052 and 5201
# Run it from a folder only administrators can change (e.g. the checkout in your own profile), never from a folder
# every user can write to (such as the Public profile): an elevated script there can be replaced by any local user.
# setup_usb4_link.ps1 -PublishPorts re-adds the portproxy entries. Parameters: see relay_user.ps1.
param(
  [string]$ListenAddress = '10.77.0.1',
  [Alias('Port')][string[]]$Ports = @('50052', '5201'),
  [string]$WslIp = '',
  [int]$BridgePort = 0,
  [string]$Distro = ''
)
$ErrorActionPreference = 'Stop'
$relay = Join-Path $PSScriptRoot 'relay_user.ps1'
if (-not (Test-Path -LiteralPath $relay)) { throw "relay_user.ps1 not found next to this script ($PSScriptRoot)" }
foreach ($p in ($Ports -join ',') -split '[,\s]+') {
  if ($p) { netsh interface portproxy delete v4tov4 listenaddress=$ListenAddress listenport=$p | Out-Null }
}
Start-Sleep 1
& $relay @PSBoundParameters
