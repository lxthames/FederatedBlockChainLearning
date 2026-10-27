param([switch]$Worker)
$ErrorActionPreference = 'Stop'
$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experimentPython = Join-Path $experimentRoot '.venv-trust\Scripts\python.exe'
$experimentJob = Join-Path $experimentRoot 'artifacts\lifecycle-v25'
$trustJob = Join-Path $experimentRoot 'artifacts\trust-full-20261022'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
Push-Location -LiteralPath $experimentRoot
try {
    if (-not $Worker) {
        $existing = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'scripts\.trust_lifecycle_v25\.study.*--run'
        })
        if ($existing.Count) { throw 'Lifecycle V25 is already running.' }
        & $experimentPython -m scripts.trust_lifecycle_v25.study --freeze
        if ($LASTEXITCODE -ne 0) { throw 'Lifecycle freeze failed.' }
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $supervisor = Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSCommandPath`"",'-Worker') `
            -WorkingDirectory $experimentRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $experimentJob "supervisor-$stamp.stdout.log") `
            -RedirectStandardError (Join-Path $experimentJob "supervisor-$stamp.stderr.log")
        [ordered]@{supervisor_pid=$supervisor.Id;started_utc=(Get-Date).ToUniversalTime().ToString('o');total=400} | ConvertTo-Json |
            Set-Content -LiteralPath (Join-Path $experimentJob 'launcher.json') -Encoding UTF8
        Write-Output "Lifecycle supervisor started: $($supervisor.Id)"
        return
    }
    $trustWorkers = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'scripts\.research\.workflow\s+run\s+--study-name\s+trust-full-20261022(?:\s|$)'
    })
    $resumeTrust = $trustWorkers.Count -gt 0
    if ($resumeTrust) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        Copy-Item -LiteralPath (Join-Path $trustJob 'status.json') -Destination (Join-Path $trustJob "status-before-lifecycle-$stamp.json")
        foreach ($workerProcess in $trustWorkers) {
            Stop-Process -Id $workerProcess.ProcessId -Force -ErrorAction SilentlyContinue
        }
        foreach ($workerProcess in $trustWorkers) {
            Wait-Process -Id $workerProcess.ProcessId -Timeout 30 -ErrorAction SilentlyContinue
        }
        $trustState = Get-Content -LiteralPath (Join-Path $trustJob 'status.json') -Raw | ConvertFrom-Json
        $trustState.status = 'queued_after_lifecycle_v25'
        $trustState.updated_utc = (Get-Date).ToUniversalTime().ToString('o')
        $trustState | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $trustJob 'status.json') -Encoding UTF8
    }
    try {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $stdout = Join-Path $experimentJob "worker-$stamp.stdout.log"
        $stderr = Join-Path $experimentJob "worker-$stamp.stderr.log"
        $lifecycle = Start-Process -FilePath $experimentPython `
            -ArgumentList @('-u','-m','scripts.trust_lifecycle_v25.study','--run') `
            -WorkingDirectory $experimentRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        [ordered]@{status='lifecycle_running';lifecycle_pid=$lifecycle.Id;resume_trust=$resumeTrust;stdout=$stdout;stderr=$stderr} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentJob 'scheduler.json') -Encoding UTF8
        $lifecycle.WaitForExit()
    } finally {
        if ($resumeTrust) { & (Join-Path $PSScriptRoot 'run_trust_full.ps1') }
        [ordered]@{status='lifecycle_exited';trust_resume_requested=$resumeTrust;updated_utc=(Get-Date).ToUniversalTime().ToString('o')} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentJob 'scheduler-finished.json') -Encoding UTF8
    }
} finally { Pop-Location }
