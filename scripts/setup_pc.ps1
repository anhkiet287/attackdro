<#
  AttackDRO - Windows PC setup (Stage 1)
  Run in an ELEVATED PowerShell (Right-click PowerShell -> Run as administrator):

      cd "C:\Users\ADMIN\Documents\Claude\Projects\AttackDRO"
      Set-ExecutionPolicy -Scope Process Bypass -Force
      .\scripts\setup_pc.ps1

  This installs WSL2 + Ubuntu and checks the NVIDIA driver. After it finishes
  (and you reboot if asked), open "Ubuntu" from the Start menu and run Stage 2:
      bash scripts/setup_wsl.sh
#>

Write-Host "==> AttackDRO PC setup (Stage 1)" -ForegroundColor Cyan

# 1. NVIDIA driver check (the Windows driver is what exposes the GPU to WSL2)
Write-Host "`n[1/2] Checking NVIDIA driver..." -ForegroundColor Yellow
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    nvidia-smi
} else {
    Write-Host "  [!] nvidia-smi not found. Install the latest NVIDIA driver for the" -ForegroundColor Red
    Write-Host "      RTX 5070 Ti from https://www.nvidia.com/Download/index.aspx" -ForegroundColor Red
    Write-Host "      (or the NVIDIA App), reboot, then re-run this script." -ForegroundColor Red
}

# 2. WSL2 + Ubuntu
Write-Host "`n[2/2] Installing WSL2 + Ubuntu..." -ForegroundColor Yellow
$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if ($wsl -and (wsl -l -q 2>$null) -match 'Ubuntu') {
    Write-Host "  Ubuntu already installed. Updating WSL..." -ForegroundColor Green
    wsl --update
} else {
    wsl --install -d Ubuntu
    Write-Host "`n  >>> A REBOOT may be required. After reboot, launch 'Ubuntu' from the" -ForegroundColor Magenta
    Write-Host "  >>> Start menu, create your Linux username/password, then run:" -ForegroundColor Magenta
    Write-Host "  >>>     bash scripts/setup_wsl.sh" -ForegroundColor Magenta
}

Write-Host "`n==> Stage 1 done." -ForegroundColor Cyan
