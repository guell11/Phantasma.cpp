param(
    [string[]] $Configs = @("Release", "RelWithDebInfo")
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildDir = Join-Path $root "build-cuda-vs"
$cachePath = Join-Path $buildDir "CMakeCache.txt"
$launcherPath = Join-Path $root "scripts\run-gemma4-26b-a4b.bat"
$mainModel = Join-Path $root "Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
$draftModel = Join-Path $root "mtp-gemma-4-26B-A4B-it.gguf"

$failures = New-Object System.Collections.Generic.List[string]

function Require-Path {
    param(
        [string] $Path,
        [string] $Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        $failures.Add("$Label missing: $Path")
        return $false
    }

    return $true
}

function Require-CacheValue {
    param(
        [string[]] $Lines,
        [string] $Pattern,
        [string] $Label
    )

    if (-not ($Lines | Select-String -Pattern $Pattern -Quiet)) {
        $failures.Add("$Label not confirmed in $cachePath")
    }
}

Write-Host "phantasma.cpp Gemma 4 Windows delivery verification"
Write-Host "Root: $root"
Write-Host ""

$cacheExists = Require-Path $cachePath "CUDA CMake cache"
if ($cacheExists) {
    $cacheLines = Get-Content -LiteralPath $cachePath
    Require-CacheValue $cacheLines '^GGML_CUDA:BOOL=ON$' "GGML_CUDA=ON"
    Require-CacheValue $cacheLines '^CMAKE_CUDA_ARCHITECTURES:[^=]*=89$' "SM89 CUDA architecture"
    Require-CacheValue $cacheLines '^CMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022$' "Visual Studio 2022 generator"
}

$launcherExists = Require-Path $launcherPath "Gemma 4 launcher"
Require-Path $mainModel "Gemma 4 main model" | Out-Null
Require-Path $draftModel "Gemma 4 MTP draft model" | Out-Null

if ($launcherExists) {
    $launcher = Get-Content -LiteralPath $launcherPath -Raw
    $requiredLauncherTokens = @(
        'set "APP=server"',
        'set "BUILD_CONFIG=Release"',
        '--spec-draft-model',
        '--spec-type %SPEC_TYPE%',
        '--spec-draft-n-max',
        '--spec-draft-p-split',
        '--spec-draft-ngl',
        '--kv-unified',
        '--fit on',
        '--temp %TEMPERATURE%',
        '-np %PARALLEL%',
        '--cpu-moe',
        '-fa on',
        '--no-mmap',
        '--jinja'
    )

    foreach ($token in $requiredLauncherTokens) {
        if (-not $launcher.Contains($token)) {
            $failures.Add("launcher token missing: $token")
        }
    }

    $smokeOutput = & $launcherPath "--check-only" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $failures.Add("launcher --check-only smoke failed with exit code $LASTEXITCODE")
    } elseif (-not (($smokeOutput -join "`n").Contains("phantasma.cpp launcher check OK"))) {
        $failures.Add("launcher --check-only smoke did not report success")
    }
}

$rows = foreach ($config in $Configs) {
    $binDir = Join-Path $buildDir "bin\$config"
    $core = Join-Path $binDir "llama.dll"
    $cli = Join-Path $binDir "llama-cli.exe"
    $server = Join-Path $binDir "llama-server.exe"
    $cuda = Join-Path $binDir "ggml-cuda.dll"

    [pscustomobject]@{
        Config = $config
        Core = Test-Path -LiteralPath $core -PathType Leaf
        CLI = Test-Path -LiteralPath $cli -PathType Leaf
        Server = Test-Path -LiteralPath $server -PathType Leaf
        CUDA = Test-Path -LiteralPath $cuda -PathType Leaf
    }
}

$rows | Format-Table -AutoSize

foreach ($row in $rows) {
    foreach ($field in @("Core", "CLI", "Server", "CUDA")) {
        if (-not $row.$field) {
            $failures.Add("$($row.Config) $field artifact missing")
        }
    }
}

if ($failures.Count -ne 0) {
    Write-Host ""
    Write-Host "Delivery verification FAILED:"
    foreach ($failure in $failures) {
        Write-Host "  - $failure"
    }
    exit 1
}

Write-Host ""
Write-Host "Delivery verification PASSED."
exit 0
