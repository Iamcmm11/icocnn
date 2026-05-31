param(
    [ValidateSet("csim", "synth", "cosim", "all")]
    [string]$Mode = "synth",
    [int]$PollSeconds = 30,
    [int]$TimeoutMinutes = 0
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$LogDir = Join-Path $Root "hls_eval_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutLog = Join-Path $LogDir "vitis_hls_${Mode}_${Stamp}.stdout.log"
$StderrLog = Join-Path $LogDir "vitis_hls_${Mode}_${Stamp}.stderr.log"
$SummaryLog = Join-Path $LogDir "vitis_hls_${Mode}_${Stamp}.summary.log"
$VitisWorkLog = Join-Path $Root "vitis_hls.log"

$VitisCandidates = @()
$VitisRun = Get-Command vitis-run.bat -ErrorAction SilentlyContinue
if ($VitisRun) {
    $VitisRoot = Split-Path -Parent (Split-Path -Parent $VitisRun.Source)
    $VitisCandidates += Join-Path $VitisRoot "bin\unwrapped\win64.o\vitis_hls.exe"
}
$VitisHlsBat = Get-Command vitis_hls.bat -ErrorAction SilentlyContinue
if ($VitisHlsBat) {
    $VitisHlsRoot = Split-Path -Parent (Split-Path -Parent $VitisHlsBat.Source)
    $VitisCandidates += Join-Path $VitisHlsRoot "bin\unwrapped\win64.o\vitis_hls.exe"
    $VitisCandidates += $VitisHlsBat.Source
}
$VitisHls = Get-Command vitis_hls -ErrorAction SilentlyContinue
if ($VitisHls) {
    $VitisCandidates += $VitisHls.Source
}

$VitisPath = $VitisCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $VitisPath) {
    throw "vitis_hls was not found in PATH."
}

$Args = @("-f", "run_hls.tcl", $Mode)
$Start = Get-Date

"Start: $Start" | Tee-Object -FilePath $SummaryLog
"Command: $VitisPath $($Args -join ' ')" | Tee-Object -FilePath $SummaryLog -Append
"Stdout: $StdoutLog" | Tee-Object -FilePath $SummaryLog -Append
"Stderr: $StderrLog" | Tee-Object -FilePath $SummaryLog -Append
"VitisWorkLog: $VitisWorkLog" | Tee-Object -FilePath $SummaryLog -Append
"PollSeconds: $PollSeconds" | Tee-Object -FilePath $SummaryLog -Append
"TimeoutMinutes: $TimeoutMinutes" | Tee-Object -FilePath $SummaryLog -Append

$Process = Start-Process -FilePath $VitisPath `
    -ArgumentList $Args `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -NoNewWindow `
    -PassThru

try {
    while (-not $Process.HasExited) {
        Start-Sleep -Seconds $PollSeconds
        $Process.Refresh()

        $Elapsed = (Get-Date) - $Start
        $StdoutSize = if (Test-Path $StdoutLog) { (Get-Item $StdoutLog).Length } else { 0 }
        $StderrSize = if (Test-Path $StderrLog) { (Get-Item $StderrLog).Length } else { 0 }
        $ReportDir = Join-Path $Root "stage1_ifan_c8_r2_hls_prj\sol1\syn\report"
        $CsynthRpt = Join-Path $ReportDir "ifan_stage1_top_csynth.rpt"
        $SizeRpt = Join-Path $ReportDir "csynth_design_size.rpt"

        $Line = "[{0:yyyy-MM-dd HH:mm:ss}] elapsed={1:hh\:mm\:ss} pid={2} cpu={3:N1}s ws={4:N1}MB stdout={5} stderr={6}" -f `
            (Get-Date), $Elapsed, $Process.Id, $Process.CPU, ($Process.WorkingSet64 / 1MB), $StdoutSize, $StderrSize
        $Line | Tee-Object -FilePath $SummaryLog -Append

        if (Test-Path $CsynthRpt) {
            "Report ready: $CsynthRpt" | Tee-Object -FilePath $SummaryLog -Append
        } elseif (Test-Path $SizeRpt) {
            "Design-size report present: $SizeRpt" | Tee-Object -FilePath $SummaryLog -Append
            Get-Content -Path $SizeRpt -Tail 20 | Tee-Object -FilePath $SummaryLog -Append
        }

        if (Test-Path $VitisWorkLog) {
            "--- vitis_hls.log tail ---" | Tee-Object -FilePath $SummaryLog -Append
            Get-Content -Path $VitisWorkLog -Tail 30 | Tee-Object -FilePath $SummaryLog -Append
        }
        if (Test-Path $StdoutLog) {
            "--- stdout tail ---" | Tee-Object -FilePath $SummaryLog -Append
            Get-Content -Path $StdoutLog -Tail 20 | Tee-Object -FilePath $SummaryLog -Append
        }
        if (Test-Path $StderrLog) {
            $ErrItem = Get-Item $StderrLog
            if ($ErrItem.Length -gt 0) {
                "--- stderr tail ---" | Tee-Object -FilePath $SummaryLog -Append
                Get-Content -Path $StderrLog -Tail 20 | Tee-Object -FilePath $SummaryLog -Append
            }
        }

        if ($TimeoutMinutes -gt 0 -and $Elapsed.TotalMinutes -ge $TimeoutMinutes) {
            "Timeout reached; stopping PID $($Process.Id)." | Tee-Object -FilePath $SummaryLog -Append
            Stop-Process -Id $Process.Id -Force
            throw "HLS evaluation timed out after $TimeoutMinutes minutes."
        }
    }
} finally {
    $Process.Refresh()
    $End = Get-Date
    "End: $End" | Tee-Object -FilePath $SummaryLog -Append
    "ExitCode: $($Process.ExitCode)" | Tee-Object -FilePath $SummaryLog -Append
}

$FinalReport = Join-Path $Root "stage1_ifan_c8_r2_hls_prj\sol1\syn\report\ifan_stage1_top_csynth.rpt"
$SizeReport = Join-Path $Root "stage1_ifan_c8_r2_hls_prj\sol1\syn\report\csynth_design_size.rpt"
if (Test-Path $FinalReport) {
    "Final report: $FinalReport" | Tee-Object -FilePath $SummaryLog -Append
} elseif (Test-Path $SizeReport) {
    "C-synthesis did not finish, but design-size report exists: $SizeReport" | Tee-Object -FilePath $SummaryLog -Append
}

exit $Process.ExitCode
