param(
    [string]$Participant = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$SpecPath = Join-Path $Root "bundle_spec.json"
$Spec = Get-Content -LiteralPath $SpecPath -Raw | ConvertFrom-Json
$ParticipantId = if ($Participant) { $Participant } else { $Spec.participant_default }
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Requirements = Join-Path $Root "requirements-team-matrix.txt"
$BootstrapLog = Join-Path $Root "bootstrap.log"
$StartedAtUtc = (Get-Date).ToUniversalTime()
$ExpectedRunRoot = if ($OutputRoot) { $OutputRoot } elseif ($Spec.default_run_root) { Join-Path $Root $Spec.default_run_root } else { Join-Path $Root "team_runs" }

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $BootstrapLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $Arguments"
    }
}

function Write-BootstrapFailure {
    param([System.Management.Automation.ErrorRecord]$Failure)
    $SafeParticipant = ($ParticipantId -replace '[^A-Za-z0-9._-]+', '-').Trim('-','.')
    if (-not $SafeParticipant) { $SafeParticipant = "member" }
    $FailureRoot = Join-Path $Root ".bootstrap-failure"
    New-Item -ItemType Directory -Path $FailureRoot -Force | Out-Null
    Copy-Item -LiteralPath $SpecPath -Destination (Join-Path $FailureRoot "bundle_spec.json") -Force
    if (Test-Path -LiteralPath $BootstrapLog) {
        Copy-Item -LiteralPath $BootstrapLog -Destination (Join-Path $FailureRoot "bootstrap.log") -Force
    }
    $FailureRecord = [ordered]@{
        schema_version = "1.0"
        status = "bootstrap_failed"
        bundle_id = $Spec.bundle_id
        participant = $SafeParticipant
        message = $Failure.Exception.Message
    }
    $FailureRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $FailureRoot "failure.json") -Encoding UTF8
    $Archive = Join-Path $Root ("FAILURE_RETURN_BOOTSTRAP_" + $Spec.bundle_id + "_" + $SafeParticipant + ".zip")
    Compress-Archive -Path (Join-Path $FailureRoot "*") -DestinationPath $Archive -CompressionLevel Optimal -Force
    $Hash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Output ("FAILURE_RETURN_READY path=" + $Archive + " sha256=" + $Hash)
}

try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($PyLauncher) {
            Invoke-Checked $PyLauncher.Source @("-3.11", "-m", "venv", ".venv")
        } else {
            $Python = Get-Command python.exe -ErrorAction Stop
            $Version = & $Python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($Version -ne "3.11") {
                throw "Python 3.11 is required. Found Python $Version."
            }
            Invoke-Checked $Python.Source @("-m", "venv", ".venv")
        }
    }

    if (-not $env:HF_ENDPOINT) {
        $env:HF_ENDPOINT = if ($Spec.requires_hf_token) { "https://huggingface.co" } else { "https://hf-mirror.com" }
    }
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", $Requirements)
    Invoke-Checked $VenvPython @(
        "-m", "scripts.run_team_model_pair",
        "--spec", $SpecPath,
        "verify-bundle"
    )

    $RunArguments = @(
        "-m", "scripts.run_team_model_pair",
        "--spec", $SpecPath,
        "run",
        "--participant", $ParticipantId
    )
    if ($OutputRoot) {
        $RunArguments += @("--output-root", $OutputRoot)
    }
    Invoke-Checked $VenvPython $RunArguments

    $SuccessArchive = Get-ChildItem -LiteralPath $ExpectedRunRoot -Filter "SUCCESS_RETURN_*.zip" -Recurse | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if (-not $SuccessArchive) {
        throw "Runner exited without a SUCCESS_RETURN archive."
    }
    Invoke-Checked $VenvPython @(
        "-m", "scripts.run_team_model_pair",
        "--spec", $SpecPath,
        "verify-return",
        "--archive", $SuccessArchive.FullName
    )
    $Hash = (Get-FileHash -LiteralPath $SuccessArchive.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Output ("RETURN_READY path=" + $SuccessArchive.FullName + " sha256=" + $Hash)
} catch {
    $FailureSearchRoots = @($Root, $ExpectedRunRoot) | Select-Object -Unique | Where-Object { Test-Path -LiteralPath $_ }
    $CurrentFailure = $FailureSearchRoots | ForEach-Object { Get-ChildItem -LiteralPath $_ -Filter "FAILURE_RETURN_*.zip" -Recurse -ErrorAction SilentlyContinue } |
        Where-Object { $_.LastWriteTimeUtc -ge $StartedAtUtc } |
        Select-Object -First 1
    if (-not $CurrentFailure) {
        Write-BootstrapFailure $_
    }
    throw
}
