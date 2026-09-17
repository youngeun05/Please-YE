#requires -Version 5.1
<#
  Environment check for the Please-YE KITTI pipeline on this PC.
  Run from anywhere:  powershell -ExecutionPolicy Bypass -File .\scripts\check_env.ps1
#>
[CmdletBinding()]
param(
    [string]$DatasetRoot = (Join-Path $env:USERPROFILE 'Documents\datasets\KITTI'),
    [string]$ZipDir      = (Join-Path $env:USERPROFILE 'Downloads')
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Section($name) { Write-Host ""; Write-Host "== $name" -ForegroundColor Cyan }

Section 'Repository'
Write-Host "repo root   : $repoRoot"
Write-Host "dataset root: $DatasetRoot"
Write-Host "zip dir     : $ZipDir"
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Push-Location $repoRoot
    Write-Host "git HEAD    : $(git rev-parse --short HEAD 2>$null) $(git log -1 --pretty=%s 2>$null)"
    Write-Host "git branch  : $(git rev-parse --abbrev-ref HEAD 2>$null)"
    Pop-Location
} else {
    Write-Host "git         : NOT FOUND" -ForegroundColor Yellow
}

Section 'Python'
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if ($py) {
    Write-Host "python path : $($py.Source)"
    & $py.Source -c "import sys; print('python ver  :', sys.version.split()[0])"
} else {
    Write-Host "python      : NOT FOUND  -> install Python 3.11+ first" -ForegroundColor Yellow
}

Section 'PyTorch / CUDA'
if ($py) {
@'
try:
    import torch
    print("torch       :", torch.__version__)
    print("cuda build  :", torch.version.cuda)
    print("cuda avail  :", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu[{i}]    : {p.name}  {p.total_memory/1024**3:.1f} GB")
except ImportError:
    print("torch       : NOT INSTALLED")
try:
    import ultralytics
    print("ultralytics :", ultralytics.__version__)
except ImportError:
    print("ultralytics : NOT INSTALLED")
for mod in ("numpy", "PIL", "yaml"):
    try:
        m = __import__(mod)
        print(f"{mod:<12}: {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"{mod:<12}: NOT INSTALLED")
'@ | & $py.Source -
}

Section 'NVIDIA driver'
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} else {
    Write-Host "nvidia-smi  : NOT FOUND" -ForegroundColor Yellow
}

Section 'Disk space'
foreach ($p in @($repoRoot, $DatasetRoot, $ZipDir)) {
    $qualifier = try { (Split-Path -Qualifier ([IO.Path]::GetFullPath($p))) } catch { $null }
    if ($qualifier) {
        $d = Get-PSDrive -Name $qualifier.TrimEnd(':') -ErrorAction SilentlyContinue
        if ($d) { Write-Host ("{0,-3} free {1,8:N1} GB   ({2})" -f $qualifier, ($d.Free/1GB), $p) }
    }
}
Write-Host ""
Write-Host "KITTI needs about 30 GB free: 12 GB zip + 12 GB extracted PNG + YOLO labels/runs."

Section 'Dataset presence'
$img = Join-Path $DatasetRoot 'training\image_2'
$lbl = Join-Path $DatasetRoot 'training\label_2'
foreach ($pair in @(@{n='images'; p=$img; e='*.png'}, @{n='labels'; p=$lbl; e='*.txt'})) {
    if (Test-Path $pair.p) {
        $n = (Get-ChildItem -LiteralPath $pair.p -Filter $pair.e -File -ErrorAction SilentlyContinue).Count
        $mark = if ($n -eq 7481) { 'OK' } else { 'INCOMPLETE' }
        Write-Host ("{0,-7}: {1} files  [{2}]  {3}" -f $pair.n, $n, $mark, $pair.p)
    } else {
        Write-Host ("{0,-7}: missing  {1}" -f $pair.n, $pair.p)
    }
}
foreach ($z in @('data_object_image_2.zip', 'data_object_label_2.zip')) {
    $zp = Join-Path $ZipDir $z
    if (Test-Path $zp) {
        Write-Host ("zip    : {0}  {1:N2} GB" -f $z, ((Get-Item $zp).Length/1GB))
    } else {
        Write-Host ("zip    : {0}  missing" -f $z)
    }
}

Section 'Committed splits'
foreach ($f in @('splits\train.txt','splits\internal_val.txt','splits\calibration.txt','eval_val.txt')) {
    $p = Join-Path $repoRoot $f
    if (Test-Path $p) {
        $n = (Get-Content -LiteralPath $p | Where-Object { $_.Trim() }).Count
        Write-Host ("{0,-28}: {1} ids" -f $f, $n)
    } else {
        Write-Host ("{0,-28}: MISSING" -f $f) -ForegroundColor Yellow
    }
}
Write-Host ""
