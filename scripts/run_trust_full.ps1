param([string]$StudyName = 'trust-full-20261022')
$ErrorActionPreference = 'Stop'
if ($StudyName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid study name.' }
$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experimentPython = Join-Path $experimentRoot '.venv-trust\Scripts\python.exe'
$experimentJob = Join-Path $experimentRoot "artifacts\$StudyName"
$existingWorkers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'scripts\.research\.workflow.*run'
})
if ($existingWorkers.Count -gt 0) { throw 'A TRUST workflow is already running.' }
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
Push-Location -LiteralPath $experimentRoot
try {
    & $experimentPython -m scripts.research.workflow prepare --study-name $StudyName
    if ($LASTEXITCODE -ne 0) { throw 'Preparation or frozen-input validation failed.' }
    $experimentStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $experimentStdout = Join-Path $experimentJob "worker-$experimentStamp.stdout.log"
    $experimentStderr = Join-Path $experimentJob "worker-$experimentStamp.stderr.log"
    $experimentWorker = Start-Process -FilePath $experimentPython `
        -ArgumentList @('-u','-m','scripts.research.workflow','run','--study-name',$StudyName) `
        -WorkingDirectory $experimentRoot -WindowStyle Hidden `
        -RedirectStandardOutput $experimentStdout -RedirectStandardError $experimentStderr -PassThru
    $launch = [ordered]@{
        pid = $experimentWorker.Id
        started_utc = (Get-Date).ToUniversalTime().ToString('o')
        study = $StudyName
        stdout = $experimentStdout
        stderr = $experimentStderr
        status = (Join-Path $experimentJob 'status.json')
        freeze_sha256 = (Get-FileHash -LiteralPath (Join-Path $experimentJob 'frozen-run.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentJob 'launcher.json') -Encoding UTF8
    $launch | ConvertTo-Json
} finally { Pop-Location }
