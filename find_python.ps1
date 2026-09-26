$ErrorActionPreference = "SilentlyContinue"

$candidates = [System.Collections.Generic.List[string]]::new()

foreach ($commandName in @("python.exe", "python3.exe", "python3.13.exe", "python313.exe")) {
    $command = Get-Command $commandName -CommandType Application | Select-Object -First 1
    if ($command -and $command.Source) {
        $candidates.Add($command.Source)
    }
}

foreach ($version in @("3.13", "3.12", "3.11", "3.10")) {
    foreach ($root in @(
        "Registry::HKEY_CURRENT_USER\Software\Python\PythonCore",
        "Registry::HKEY_LOCAL_MACHINE\Software\Python\PythonCore",
        "Registry::HKEY_LOCAL_MACHINE\Software\WOW6432Node\Python\PythonCore"
    )) {
        $installKey = Join-Path $root "$version\InstallPath"
        if (Test-Path $installKey) {
            $key = Get-Item $installKey
            $executable = $key.GetValue("ExecutablePath")
            $directory = $key.GetValue("")
            if ($executable) {
                $candidates.Add([string]$executable)
            }
            if ($directory) {
                $candidates.Add((Join-Path ([string]$directory) "python.exe"))
            }
        }
    }
}

foreach ($path in @(
    "$env:LOCALAPPDATA\Python\bin\python.exe",
    "$env:LOCALAPPDATA\Python\bin\python3.exe",
    "$env:LOCALAPPDATA\Python\bin\python3.13.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:ProgramFiles\Python313\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "$env:ProgramFiles\Python311\python.exe",
    "$env:ProgramFiles\Python310\python.exe",
    "$env:SystemDrive\Python313\python.exe",
    "$env:SystemDrive\Python312\python.exe"
)) {
    $candidates.Add($path)
}

foreach ($managerPath in @(
    "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.PythonManager_3847v3x7pw1km\py.exe",
    "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.PythonManager_qbz5n2kfra8p0\py.exe"
)) {
    if (Test-Path -LiteralPath $managerPath -PathType Leaf) {
        $managedExecutable = & $managerPath -3.13 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $managedExecutable) {
            $candidates.Add([string]$managedExecutable)
        }
    }
}

$storePackage = Get-AppxPackage -Name "PythonSoftwareFoundation.Python.3.13" | Select-Object -First 1
if ($storePackage -and $storePackage.InstallLocation) {
    $candidates.Add((Join-Path $storePackage.InstallLocation "python.exe"))
}

foreach ($candidate in $candidates | Select-Object -Unique) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        continue
    }

    $result = & $candidate -c "import struct, sys; print('compatible' if (3, 10) <= sys.version_info[:2] <= (3, 13) and struct.calcsize('P') == 8 else 'incompatible')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $result -eq "compatible") {
        Write-Output $candidate
        exit 0
    }
}

exit 1
