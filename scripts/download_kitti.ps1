#requires -Version 5.1
<#
  Download the KITTI 2D object detection archives (left color images + labels).

  These are the archives the pipeline expects:
    data_object_image_2.zip   about 12 GB
    data_object_label_2.zip   about 5 MB

  The download is resumable: re-run the script if it is interrupted.
  If the mirror below refuses the request, register at
  https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
  and download the same two files manually into -ZipDir.
#>
[CmdletBinding()]
param(
    [string]$ZipDir = (Join-Path $env:USERPROFILE 'Downloads'),
    [string]$BaseUrl = 'https://s3.eu-central-1.amazonaws.com/avg-kitti'
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $ZipDir | Out-Null

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue

foreach ($name in @('data_object_label_2.zip', 'data_object_image_2.zip')) {
    $dest = Join-Path $ZipDir $name
    $url  = "$BaseUrl/$name"
    Write-Host ""
    Write-Host "== $name" -ForegroundColor Cyan
    Write-Host "   from $url"
    Write-Host "   to   $dest"

    if ($curl) {
        # -C - resumes a partial file, -L follows redirects.
        & $curl.Source -L -C - --retry 5 --retry-delay 5 -o $dest $url
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 33) {
            throw "curl failed for $name (exit $LASTEXITCODE)"
        }
    } else {
        Start-BitsTransfer -Source $url -Destination $dest -Description $name
    }

    $size = (Get-Item -LiteralPath $dest).Length
    Write-Host ("   done: {0:N2} GB" -f ($size / 1GB))
}

Write-Host ""
Write-Host "Next: .\scripts\run_prepare.ps1 -ZipDir `"$ZipDir`"" -ForegroundColor Green
