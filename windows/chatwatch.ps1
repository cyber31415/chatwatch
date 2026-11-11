# chatwatch for Windows: sets up WSL2's Ubuntu the first time, then starts chatwatch in it and opens the page.
# Double-click "Start chatwatch.cmd" beside this file, or in PowerShell:
#   powershell -ExecutionPolicy Bypass -File chatwatch.ps1 [-NoBrowser] [-From <folder>]
# What it does (each step is skipped once done, so later runs just start chatwatch):
#   1. gives WSL enough memory for the AI models in %USERPROFILE%\.wslconfig (your RAM minus 3 GB; a copy of the old file
#      is kept) and restarts WSL, after asking
#   2. installs Ubuntu 24.04 in WSL (it asks you for a new username and password)
#   3. inside Ubuntu (setup_wsl.sh): podman, and chatwatch itself from GitHub (this version)
#   4. starts chatwatch and opens http://127.0.0.1:8765. Ctrl+C stops it, and then stops Ubuntu (and WSL, when nothing
#      else runs in it), so the memory goes back to Windows
# -From <folder>: copy chatwatch from a folder instead of GitHub (testing a version not yet released).
param([switch]$NoBrowser, [string]$From = "")
$Version = "0.9.1-beta"                 # the GitHub release this launcher belongs to (the repository's VERSION file)
$Distro = "Ubuntu-24.04"
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"
# wsl.exe (with WSL_UTF8) and the Linux programs it runs answer in UTF-8; Windows PowerShell 5.1 would read them in the
# console's old code page (a user folder with an accented name would come back garbled)
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }
$Here = $PSScriptRoot
$Work = Join-Path $env:USERPROFILE ".chatwatch"
$Cfg = Join-Path $env:USERPROFILE ".wslconfig"
New-Item -ItemType Directory -Force -Path $Work | Out-Null
Set-Location $env:USERPROFILE          # wsl.exe cannot start in a network-share folder

function Say([string]$m) { Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host ""; Write-Host $m -ForegroundColor Red; exit 1 }
function Clean([object]$o) { (($o | Out-String) -replace "`0", "").Trim() }
function Distros([string[]]$extra) {
  @((Clean (& wsl.exe -l -q @extra)) -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -and $_ -notmatch '^There are no' })
}

Write-Host "chatwatch $Version for Windows"

# ---- 0. WSL itself
& wsl.exe --version *> $null
if ($LASTEXITCODE -ne 0) {
  Fail ("WSL is not installed. Open PowerShell as administrator, run:`n    wsl --install --no-distribution`n" +
        "then restart the PC and start chatwatch again.")
}

# ---- 1. WSL's memory: its default is half the RAM, too little for the AI models. Only ever raised, never lowered.
$totalGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$want = [math]::Max(4, $totalGB - 3)
$wantLine = "memory=${want}GB"
$lines = @(); if (Test-Path $Cfg) { $lines = [IO.File]::ReadAllLines($Cfg) }
$out = New-Object System.Collections.Generic.List[string]
$inWsl2 = $false; $seenWsl2 = $false; $done = $false
foreach ($l in $lines) {
  $t = $l.Trim()
  if ($t -match '^\[.+\]$') {
    if ($inWsl2 -and -not $done) { $out.Add($wantLine); $done = $true }
    $inWsl2 = ($t -ieq '[wsl2]'); if ($inWsl2) { $seenWsl2 = $true }
    $out.Add($l); continue
  }
  if ($inWsl2 -and $t -match '^memory\s*=\s*(\d+)\s*([GgMm][Bb])?') {
    $cur = [int]$Matches[1]; if ($Matches[2] -imatch 'mb') { $cur = [math]::Floor($cur / 1024) }
    if ($cur -ge $want) { $out.Add($l) } else { $out.Add($wantLine) }
    $done = $true; continue
  }
  $out.Add($l)
}
if ($inWsl2 -and -not $done) { $out.Add($wantLine); $done = $true }
if (-not $seenWsl2) { $out.Add("[wsl2]"); $out.Add($wantLine) }
$new = ($out -join "`r`n").Trim() + "`r`n"
$old = ""; if (Test-Path $Cfg) { $old = [IO.File]::ReadAllText($Cfg) }
if ($new.Trim() -ne $old.Trim()) {
  Say "WSL needs more memory for the AI models: $want GB of your $totalGB GB (a limit, not a reservation: Windows keeps what WSL does not use). This is written to $Cfg, and WSL restarts (open Ubuntu windows close)."
  Read-Host "Press Enter to continue (or close this window to stop)" | Out-Null
  $stamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
  if (Test-Path $Cfg) { Copy-Item $Cfg "$Cfg.bak-$stamp"; Write-Host "the old file is kept as $Cfg.bak-$stamp" }
  [IO.File]::WriteAllText($Cfg, $new, (New-Object System.Text.UTF8Encoding($false)))
  & wsl.exe --shutdown
}

# ---- 2. Ubuntu 24.04
if ((Distros @()) -notcontains $Distro) {
  Say "Installing $Distro. When it asks, choose a username and a password (chatwatch's setup asks for that password next). If it then shows an Ubuntu prompt (ending in `$), type  exit  and press Enter."
  & wsl.exe --install -d $Distro
  if ((Distros @()) -notcontains $Distro) { Fail "$Distro did not finish installing: start chatwatch again." }
}

# ---- 3. inside Ubuntu: the Linux half of the setup, copied where Ubuntu can read it, with Linux line endings
$src = Join-Path $Here "setup_wsl.sh"
if (-not (Test-Path $src)) { Fail "setup_wsl.sh is missing beside chatwatch.ps1: extract the whole zip again." }
$dst = Join-Path $Work "setup_wsl.sh"
[IO.File]::WriteAllText($dst, ([IO.File]::ReadAllText($src) -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
$LinuxScript = Clean (& wsl.exe -d $Distro -e wslpath -a $dst)
$FromArg = @()
if ($From) {
  # a test copy: without what chatwatch downloads or writes itself
  if (-not (Test-Path (Join-Path $From "start.sh"))) { Fail "no chatwatch in $From" }
  $copy = Join-Path $Work "from"
  $skip = @(".git", "models\ollama", "models\unbiased-toxic-roberta-onnx", "models\all-MiniLM-L6-v2", "data\state",
            "data\sessions", "data\emotes", "data\mlcache", "data\raw", ".pytest_cache") | ForEach-Object { Join-Path $From $_ }
  & robocopy.exe $From $copy /MIR /XD @skip /XF "*.pyc" /NFL /NDL /NJH /NJS /NP | Out-Null
  if ($LASTEXITCODE -ge 8) { Fail "the copy from $From failed (robocopy code $LASTEXITCODE)" }
  $FromArg = @(Clean (& wsl.exe -d $Distro -e wslpath -a $copy))
}
& wsl.exe -d $Distro -e bash $LinuxScript setup $Version @FromArg
if ($LASTEXITCODE -eq 3) {
  Say "Restarting $Distro (systemd turned on)"
  & wsl.exe --terminate $Distro
  & wsl.exe -d $Distro -e bash $LinuxScript setup $Version @FromArg
}
if ($LASTEXITCODE -ne 0) { Fail "The setup stopped (see the lines above). Fix what it says and start chatwatch again." }

# ---- 4. chatwatch, and the page once it answers (the first start builds chatwatch's image: a few minutes)
$browser = $null
if (-not $NoBrowser) {
  $browser = Start-Job -ScriptBlock {
    for ($i = 0; $i -lt 300; $i++) {
      try { $c = New-Object Net.Sockets.TcpClient; $c.Connect("127.0.0.1", 8765); $c.Close(); Start-Process "http://127.0.0.1:8765"; return }
      catch { Start-Sleep -Seconds 2 }
    }
  }
}
Say "Starting chatwatch: the page opens at http://127.0.0.1:8765 when it is ready. Press Ctrl+C here to stop it."
try {
  & wsl.exe -d $Distro -e bash $LinuxScript start
} finally {
  if ($browser) { Stop-Job $browser -ErrorAction SilentlyContinue; Remove-Job $browser -Force -ErrorAction SilentlyContinue }
  Write-Host ""
  Write-Host "Stopping $Distro so Windows gets its memory back..."
  & wsl.exe --terminate $Distro | Out-Null
  $installed = Distros @()
  $running = @(Distros @("--running") | Where-Object { $installed -contains $_ })
  if ($running.Count -eq 0) { & wsl.exe --shutdown | Out-Null; Write-Host "Done: WSL is shut down and its memory is back with Windows." }
  else { Write-Host "Other WSL distributions are still running ($($running -join ', ')): WSL keeps the memory they use." }
}
