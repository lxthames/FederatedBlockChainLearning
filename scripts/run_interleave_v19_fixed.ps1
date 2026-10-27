$ErrorActionPreference = 'Stop'
$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experimentPython = Join-Path $experimentRoot '.venv-trust\Scripts\python.exe'
$experimentJob = Join-Path $experimentRoot 'artifacts\interleave-v19-fixed'
$experimentFreeze = Join-Path $experimentJob 'frozen-inputs.json'
if (-not (Test-Path -LiteralPath $experimentFreeze)) {
    throw 'Freeze V19 with python -m scripts.trust_interleave_v19_fixed.study --freeze before launching.'
}
$existingWorkers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*scripts.trust_interleave_v19_fixed.study*--run*'
})
if ($existingWorkers.Count -gt 0) {
    throw "V19 is already running: $($existingWorkers.ProcessId -join ', ')"
}
$experimentStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$experimentStdout = Join-Path $experimentJob "worker-$experimentStamp.stdout.log"
$experimentStderr = Join-Path $experimentJob "worker-$experimentStamp.stderr.log"
$experimentWorker = Start-Process -FilePath $experimentPython `
    -ArgumentList @('-u', '-m', 'scripts.trust_interleave_v19_fixed.study', '--run') `
    -WorkingDirectory $experimentRoot -WindowStyle Hidden `
    -RedirectStandardOutput $experimentStdout -RedirectStandardError $experimentStderr -PassThru
$experimentLaunch = [ordered]@{
    pid = $experimentWorker.Id
    started_utc = (Get-Date).ToUniversalTime().ToString('o')
    stdout = $experimentStdout
    stderr = $experimentStderr
    freeze_sha256 = (Get-FileHash -LiteralPath $experimentFreeze -Algorithm SHA256).Hash.ToLowerInvariant()
    launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    resume_command = '.\scripts\run_interleave_v19_fixed.ps1'
}
$experimentLaunch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentJob 'launcher.json') -Encoding UTF8
$experimentLaunch | ConvertTo-Json
