#Requires -RunAsAdministrator
# hive-ram: undo setup_usb4_link.ps1 for one link (ADMIN PowerShell): that address's firewall rules, its portproxy
# entries, the address itself, and the adapter back to DHCP with network category Public.
#   powershell -ExecutionPolicy Bypass -File undo_usb4_link.ps1                     # the one link set up on this node
#   powershell -ExecutionPolicy Bypass -File undo_usb4_link.ps1 -Address 10.78.0.1  # one link of several
#   powershell -ExecutionPolicy Bypass -File undo_usb4_link.ps1 -All                # every link
# Also removed: rules an earlier version of setup made ('N1X cluster*', with a 10.77.0.x link or -All) and relay rules
# named 'tfpp-glm-*' that are scoped to the link's address. It reports each step and exits 1 if any step failed.
param([string]$Address = '', [switch]$All)
$ErrorActionPreference = 'Stop'
$script:failed = 0

function Invoke-Step([string]$What, [scriptblock]$Do) {
  try { & $Do; "done: $What" }
  catch { $script:failed++; "FAILED: $What ($($_.Exception.Message))" }
}

function Remove-Rules($Rules, [string]$Why) {
  foreach ($r in @($Rules | Where-Object { $_ })) {
    Invoke-Step "remove firewall rule '$($r.DisplayName)'$Why" { $r | Remove-NetFirewallRule }
  }
}

# addresses this node's links use: named by hive-ram rules, or set by hand on a USB4 adapter
$byRule = @(Get-NetFirewallRule -Group 'hive-ram' -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.DisplayName -match '^hive-ram (\d+\.\d+\.\d+\.\d+) ') { $Matches[1] } })
$usb4 = @(Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceDescription -match 'USB4|P2P' })
$byAdapter = @(foreach ($ad in $usb4) {
    Get-NetIPAddress -InterfaceIndex $ad.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
      Where-Object { $_.PrefixOrigin -eq 'Manual' } | ForEach-Object { $_.IPAddress } })
$found = @($byRule + $byAdapter | Where-Object { $_ } | Sort-Object -Unique)

if ($Address) { $targets = @($Address) }
elseif ($All -or $found.Count -le 1) { $targets = $found }
else { throw "several links on this node ($($found -join ', ')): pass -Address <one of them>, or -All" }

$proxies = @((netsh interface portproxy show v4tov4) | ForEach-Object {
    if ($_ -match '^\s*(\d+\.\d+\.\d+\.\d+)\s+(\d+)\s+\S+\s+\d+\s*$') { [pscustomobject]@{ Address = $Matches[1]; Port = $Matches[2] } } })
$legacy = $All
foreach ($t in $targets) {
  "link $t"
  if ($t -like '10.77.0.*') { $legacy = $true }
  Remove-Rules (Get-NetFirewallRule -DisplayName "hive-ram $t *" -ErrorAction SilentlyContinue) ''
  Remove-Rules (Get-NetFirewallRule -DisplayName 'tfpp-glm-*' -ErrorAction SilentlyContinue | Where-Object {
      @(($_ | Get-NetFirewallAddressFilter).LocalAddress) -contains $t }) " (scoped to $t)"
  foreach ($p in @($proxies | Where-Object { $_.Address -eq $t })) {
    Invoke-Step "remove portproxy $($p.Address):$($p.Port)" {
      $out = netsh interface portproxy delete v4tov4 listenaddress=$($p.Address) listenport=$($p.Port)
      if ($LASTEXITCODE -ne 0) { throw ([string]($out -join ' ')).Trim() }
    }
  }
  $held = @(Get-NetIPAddress -IPAddress $t -AddressFamily IPv4 -ErrorAction SilentlyContinue)
  foreach ($h in $held) {
    $idx = $h.InterfaceIndex; $name = $h.InterfaceAlias
    Invoke-Step "remove $t from '$name'" { $h | Remove-NetIPAddress -Confirm:$false }
    $rest = @(Get-NetIPAddress -InterfaceIndex $idx -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.PrefixOrigin -eq 'Manual' })
    if ($rest.Count -eq 0) {
      Invoke-Step "set '$name' to network category Public" { Set-NetConnectionProfile -InterfaceIndex $idx -NetworkCategory Public }
      Invoke-Step "turn DHCP back on for '$name'" { Set-NetIPInterface -InterfaceIndex $idx -Dhcp Enabled }
    }
  }
  if (-not $held) { "  $t is not on any adapter" }
}
if ($legacy -or $targets.Count -eq 0) {
  Remove-Rules (Get-NetFirewallRule -DisplayName 'N1X cluster*' -ErrorAction SilentlyContinue) ' (earlier setup)'
}
if ($targets.Count -eq 0) { 'no hive-ram link found on this node' }
if ($script:failed) { "$($script:failed) step(s) failed, see above"; exit 1 }
'link settings removed'
