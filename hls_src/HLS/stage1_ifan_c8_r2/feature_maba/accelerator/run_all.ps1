param(
    [string]$Python = "python",
    [ValidateSet("tiling","fusion","interleave","fold","all")]
    [string]$Step = "tiling",
    [switch]$RefreshSynthesis,
    [int]$TimeoutSeconds = 900
)
$ErrorActionPreference = "Stop"
# No RTL or background launch: one mechanism at a time.
$mabaArguments = @("$PSScriptRoot/compare.py", "--step", $Step, "--verify", "--timeout", "$TimeoutSeconds")
if ($RefreshSynthesis) { $mabaArguments += "--refresh-synth" }
& $Python @mabaArguments
if ($LASTEXITCODE -ne 0) { throw "Comparison failed; later stages were not started." }
