$ErrorActionPreference = 'Stop'
$studyRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$studyPython = Join-Path $studyRoot '.venv-trust\Scripts\python.exe'
$studyJob = Join-Path $studyRoot 'artifacts\lifecycle-v25'
$studyResults = Join-Path $studyRoot 'results\lifecycle-v25'
$retryIndices = @(204,312,362)
$retryNames = @(
    'v25-fashion_mnist-finetune-continuous-patch-s4103',
    'v25-fashion_mnist-quorum-attack_stop-checkerboard-s4549',
    'v25-fashion_mnist-periodic-continuous-checkerboard-s4103'
)

function Write-RetryStatus([string]$status, [string]$current, [int]$completed, [int]$failed) {
    [ordered]@{
        status = $status
        current = $current
        completed = $completed
        failed = $failed
        total = 400
        retry_indices = $retryIndices
        updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $studyJob 'retry-status.json') -Encoding UTF8
}

Push-Location -LiteralPath $studyRoot
try {
    while ($true) {
        $activeTrust = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'scripts\.research\.workflow\s+run\s+--study-name\s+trust-full-20261022(?:\s|$)'
        })
        if ($activeTrust.Count -eq 0) { break }
        Write-RetryStatus 'waiting_for_trust_gpu' '' 397 0
        Start-Sleep -Seconds 30
    }

    & $studyPython -c "from scripts.trust_lifecycle_v25.study import check_frozen; check_frozen(check_data=True)"
    if ($LASTEXITCODE -ne 0) { throw 'Frozen lifecycle inputs no longer verify.' }

    $completed = 397
    $failed = 0
    for ($i = 0; $i -lt $retryIndices.Count; $i++) {
        $index = $retryIndices[$i]
        $name = $retryNames[$i]
        Write-RetryStatus 'retrying' $name $completed $failed
        $target = Join-Path $studyResults $name
        if (-not (Test-Path -LiteralPath $target)) { throw "Missing partial result directory: $target" }
        $archive = "$target.interrupted-retry-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
        Move-Item -LiteralPath $target -Destination $archive
        & $studyPython -u -m scripts.trust_lifecycle_v25.study --cell $index
        if ($LASTEXITCODE -ne 0) {
            $failed++
            continue
        }
        $completed++
    }

    & $studyPython -m scripts.trust_lifecycle_v25.study --report
    if ($failed -ne 0) { throw "$failed exact-code retries failed" }
    Write-RetryStatus 'complete' '' $completed 0
} catch {
    Write-RetryStatus 'failed' '' $completed ($failed + 1)
    throw
} finally {
    Pop-Location
}
