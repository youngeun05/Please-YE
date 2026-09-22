$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Destination = Join-Path $RepoRoot "third_party\mmdetection3d"
$ExpectedCommit = "fe25f7a51d36e3702f961e198894580d83c4387b"

if (-not (Test-Path -LiteralPath (Join-Path $Destination ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    git -c http.sslBackend=openssl clone https://github.com/open-mmlab/mmdetection3d.git $Destination
}

$ActualCommit = (git -C $Destination rev-parse HEAD).Trim()
if ($ActualCommit -ne $ExpectedCommit) {
    git -C $Destination fetch origin $ExpectedCommit
    git -C $Destination checkout --detach $ExpectedCommit
    $ActualCommit = (git -C $Destination rev-parse HEAD).Trim()
}
if ($ActualCommit -ne $ExpectedCommit) {
    throw "Evaluator commit mismatch: expected $ExpectedCommit, got $ActualCommit"
}

Write-Host "Pinned evaluator ready at $Destination ($ActualCommit)"
