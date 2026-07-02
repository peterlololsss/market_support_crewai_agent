$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Remote = if ($env:REMOTE) { $env:REMOTE } else { "xiaoyan@192.168.209.195" }
$RemoteRoot = "/data/xiaoyan/market_support_crewai_agent"
$RemoteApp = "$RemoteRoot/app"
$RemoteUpload = "$RemoteRoot/app.upload"
$RemoteArchive = "$RemoteRoot/app.upload.tar.gz"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$Archive = Join-Path ([System.IO.Path]::GetTempPath()) ("market-support-crewai-agent-" + [System.Guid]::NewGuid().ToString("N") + ".tar.gz")

function Die([string] $Message) {
    throw "ERROR: $Message"
}

function Need-Command([string] $Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Die "missing command: $Name"
    }
}

function Run-Native([string] $Name, [string[]] $Arguments) {
    & $Name @Arguments
    if ($LASTEXITCODE -ne 0) {
        Die "$Name exited with code $LASTEXITCODE"
    }
}

Need-Command "ssh"
Need-Command "scp"
Need-Command "tar.exe"

try {
    $preflight = "mkdir -p '$RemoteRoot' '$RemoteRoot/runtime'; test -f '$RemoteRoot/.env' || { echo 'ERROR: missing env file: $RemoteRoot/.env' >&2; exit 1; }"
    Run-Native "ssh" @($Remote, $preflight)

    Write-Host "packing current tree from $AppRoot"
    Run-Native "tar.exe" @(
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=.pytest_cache",
        "--exclude=tmp",
        "--exclude=.env",
        "--exclude=*/__pycache__",
        "-czf",
        $Archive,
        "-C",
        $AppRoot,
        "."
    )

    Write-Host "uploading archive to ${Remote}:$RemoteArchive"
    Run-Native "scp" @($Archive, "${Remote}:$RemoteArchive")

    $deploy = @(
        "set -euo pipefail",
        "[ '$RemoteApp' = '/data/xiaoyan/market_support_crewai_agent/app' ]",
        "rm -rf '$RemoteUpload'",
        "mkdir -p '$RemoteUpload' '$RemoteRoot/runtime'",
        "tar xzf '$RemoteArchive' -C '$RemoteUpload'",
        "rm -f '$RemoteArchive'",
        "test -f '$RemoteUpload/Containerfile'",
        "rm -rf '$RemoteApp'",
        "mv '$RemoteUpload' '$RemoteApp'",
        "bash '$RemoteApp/scripts/deploy_podman.sh'"
    ) -join "; "
    Run-Native "ssh" @($Remote, $deploy)
} finally {
    if (Test-Path -LiteralPath $Archive) {
        Remove-Item -LiteralPath $Archive -Force
    }
}
