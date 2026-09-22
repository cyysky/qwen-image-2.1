#!/usr/bin/env pwsh
# Start the Qwen-Image-2.1 playground. Use -Check to probe endpoints instead.
param(
    [switch]$Check,
    [switch]$CheckEnhancer,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }

$cliArgs = @("app.py")
if ($Check) { $cliArgs += "--check" }
if ($CheckEnhancer) { $cliArgs += "--check-enhancer" }
if ($Reload) { $cliArgs += "--reload" }

& $python @cliArgs
exit $LASTEXITCODE
