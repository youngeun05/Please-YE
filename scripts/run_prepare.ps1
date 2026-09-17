#requires -Version 5.1
<#
  One-shot KITTI data preparation for this PC.

  Stages:
    1. prepare_kitti.py   verify the two ZIPs and extract training/image_2 + training/label_2
    2. audit_kitti.py     check 7481 images/labels, PNG format, bbox ranges, class counts
    3. make_splits.py     (optional) regenerate splits and compare them with the committed ones
    4. junctions          data\kitti_yolo_*\images\all -> <DatasetRoot>\training\image_2
    5. convert            KITTI -> YOLO labels for exact3 and neighbors3

  Usage:
    powershell -ExecutionPolicy Bypass -File .\scripts\run_prepare.ps1
    ... -DatasetRoot D:\datasets\KITTI -ZipDir D:\downloads
    ... -VerifySplits          also regenerate splits and compare SHA256
    ... -SkipCrc               skip the full ZIP CRC pass (faster, less thorough)
#>
[CmdletBinding()]
param(
    [string]$DatasetRoot = (Join-Path $env:USERPROFILE 'Documents\datasets\KITTI'),
    [string]$ZipDir      = (Join-Path $env:USERPROFILE 'Downloads'),
    [switch]$VerifySplits,
    [switch]$SkipCrc,
    [switch]$SkipPrepare,
    [switch]$SkipAudit,
    [switch]$SkipConvert
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) { $python = (Get-Command py -ErrorAction SilentlyContinue) }
if (-not $python) { throw 'Python was not found on PATH.' }
$py = $python.Source

$imageZip = Join-Path $ZipDir 'data_object_image_2.zip'
$labelZip = Join-Path $ZipDir 'data_object_label_2.zip'
$evalSplit = Join-Path $repoRoot 'eval_val.txt'
$runs = Join-Path $repoRoot 'runs'
New-Item -ItemType Directory -Force -Path $runs | Out-Null

function Step($n, $text) { Write-Host ""; Write-Host "===== [$n] $text" -ForegroundColor Cyan }
function Invoke-Py($argList) {
    & $py @argList
    if ($LASTEXITCODE -ne 0) { throw "python $($argList[0]) failed (exit $LASTEXITCODE)" }
}

Write-Host "repo root   : $repoRoot"
Write-Host "dataset root: $DatasetRoot"
Write-Host "zip dir     : $ZipDir"

# ---------------------------------------------------------------- 1. extract
if (-not $SkipPrepare) {
    Step 1 'prepare_kitti.py - verify archives and extract'
    foreach ($z in @($imageZip, $labelZip)) {
        if (-not (Test-Path -LiteralPath $z)) {
            throw "Missing archive: $z  (run .\scripts\download_kitti.ps1 first)"
        }
    }
    $a = @('scripts\prepare_kitti.py',
           '--image-zip', $imageZip,
           '--label-zip', $labelZip,
           '--output-root', $DatasetRoot,
           '--eval-split', $evalSplit)
    if ($SkipCrc) { $a += '--skip-full-crc' }
    Invoke-Py $a
} else {
    Step 1 'prepare_kitti.py - SKIPPED'
}

# ------------------------------------------------------------------ 2. audit
if (-not $SkipAudit) {
    Step 2 'audit_kitti.py - dataset integrity report'
    Invoke-Py @('scripts\audit_kitti.py',
                '--kitti-root', $DatasetRoot,
                '--eval-split', $evalSplit,
                '--report', (Join-Path $runs 'audit_report.local.json'))
} else {
    Step 2 'audit_kitti.py - SKIPPED'
}

# ----------------------------------------------------- 3. split reproduction
if ($VerifySplits) {
    Step 3 'make_splits.py - regenerate and compare with committed splits'
    $checkDir = Join-Path $runs 'splits_check'
    New-Item -ItemType Directory -Force -Path $checkDir | Out-Null
    Invoke-Py @('scripts\make_splits.py',
                '--kitti-root', $DatasetRoot,
                '--eval-split', $evalSplit,
                '--output-dir', $checkDir)
    $mismatch = $false
    foreach ($name in @('train.txt', 'internal_val.txt', 'calibration.txt')) {
        $a1 = (Get-FileHash -LiteralPath (Join-Path $repoRoot "splits\$name") -Algorithm SHA256).Hash
        $a2 = (Get-FileHash -LiteralPath (Join-Path $checkDir $name) -Algorithm SHA256).Hash
        if ($a1 -eq $a2) {
            Write-Host ("  {0,-18} reproduced OK" -f $name) -ForegroundColor Green
        } else {
            Write-Host ("  {0,-18} MISMATCH  committed={1}  regenerated={2}" -f $name, $a1.Substring(0,12), $a2.Substring(0,12)) -ForegroundColor Red
            $mismatch = $true
        }
    }
    if ($mismatch) {
        throw 'Split regeneration does not match the committed splits. Do not proceed; the committed splits are the contract.'
    }
} else {
    Step 3 'make_splits.py - SKIPPED (committed splits are used as-is; pass -VerifySplits to re-derive)'
}

# -------------------------------------------------------------- 4. junctions
Step 4 'image directory junctions'
$rawImages = Join-Path $DatasetRoot 'training\image_2'
if (-not (Test-Path -LiteralPath $rawImages)) { throw "Missing extracted images: $rawImages" }
foreach ($variant in @('kitti_yolo_exact3', 'kitti_yolo_neighbors3')) {
    $imagesDir = Join-Path $repoRoot "data\$variant\images"
    $junction  = Join-Path $imagesDir 'all'
    New-Item -ItemType Directory -Force -Path $imagesDir | Out-Null
    if (Test-Path -LiteralPath $junction) {
        Write-Host "  exists : $junction"
    } else {
        New-Item -ItemType Junction -Path $junction -Target $rawImages | Out-Null
        Write-Host "  created: $junction -> $rawImages" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------- 5. convert
if (-not $SkipConvert) {
    Step 5 'convert_kitti_to_yolo.py - exact3 (three eval classes only)'
    Invoke-Py @('scripts\convert_kitti_to_yolo.py',
                '--kitti-root', $DatasetRoot,
                '--splits-dir', (Join-Path $repoRoot 'splits'),
                '--official-eval', $evalSplit,
                '--output-root', (Join-Path $repoRoot 'data\kitti_yolo_exact3'),
                '--neighbor-policy', 'ignore')

    Step 5 'convert_kitti_to_yolo.py - neighbors3 (Van->Car, Person_sitting->Pedestrian)'
    Invoke-Py @('scripts\convert_kitti_to_yolo.py',
                '--kitti-root', $DatasetRoot,
                '--splits-dir', (Join-Path $repoRoot 'splits'),
                '--official-eval', $evalSplit,
                '--output-root', (Join-Path $repoRoot 'data\kitti_yolo_neighbors3'),
                '--neighbor-policy', 'map')
} else {
    Step 5 'convert - SKIPPED'
}

Write-Host ""
Write-Host "===== done" -ForegroundColor Green
Write-Host "audit report : runs\audit_report.local.json"
Write-Host "datasets     : data\kitti_yolo_exact3\kitti.local.yaml"
Write-Host "               data\kitti_yolo_neighbors3\kitti.local.yaml"
Write-Host "next         : python scripts\verify_prepared.py   then   python scripts\smoke_train.py --device 0 --fraction 0.01"
