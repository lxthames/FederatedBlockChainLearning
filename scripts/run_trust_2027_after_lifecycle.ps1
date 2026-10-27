$ErrorActionPreference = 'Stop'
$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experimentPython = Join-Path $experimentRoot '.venv-trust\Scripts\python.exe'
$lifecycleStatus = Join-Path $experimentRoot 'artifacts\lifecycle-v24\status.json'
$experimentJob = Join-Path $experimentRoot 'artifacts\trust-2027'

while ($true) {
    if (Test-Path -LiteralPath $lifecycleStatus) {
        $state = Get-Content -LiteralPath $lifecycleStatus -Raw | ConvertFrom-Json
        if ($state.status -eq 'complete') { break }
        if ($state.status -eq 'failed') {
            throw "Lifecycle V24 failed; confirmatory TRUST run was not started."
        }
    }
    Start-Sleep -Seconds 30
}

$existingWorkers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*scripts.research.workflow* run*'
})
if ($existingWorkers.Count -gt 0) {
    throw "TRUST workflow is already running: $($existingWorkers.ProcessId -join ', ')"
}

$experimentStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$experimentStdout = Join-Path $experimentJob "resume-$experimentStamp.stdout.log"
$experimentStderr = Join-Path $experimentJob "resume-$experimentStamp.stderr.log"
$experimentWorker = Start-Process -FilePath $experimentPython `
    -ArgumentList @('-u', '-m', 'scripts.research.workflow', 'run') `
    -WorkingDirectory $experimentRoot -WindowStyle Hidden `
    -RedirectStandardOutput $experimentStdout -RedirectStandardError $experimentStderr -PassThru

$launch = [ordered]@{
    pid = $experimentWorker.Id
    started_utc = (Get-Date).ToUniversalTime().ToString('o')
    stdout = $experimentStdout
    stderr = $experimentStderr
    prerequisite = 'lifecycle-v24 complete'
    matrix = 'TRUST v2 workflow; resumes existing outputs and completes missing cells'
}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentJob 'resume-launcher.json') -Encoding UTF8
$launch | ConvertTo-Json
