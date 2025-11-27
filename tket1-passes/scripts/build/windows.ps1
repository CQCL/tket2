param (
    [Parameter(Mandatory = $true)]
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location (Join-Path $PSScriptRoot "..\..\..\")

choco install jq -y
pipx install conan
conan profile detect
conan remote add tket-libs `
    "https://quantinuumsw.jfrog.io/artifactory/api/conan/tket1-libs" `
    --index 0

# Run install and capture JSON output
$tmp = New-TemporaryFile
conan install tket1-passes `
    --build=missing `
    --options="tket-c-api/*:shared=True" `
    --format=json |
    Out-File -Encoding UTF8 $tmp

$libFolder = jq -r '.graph.nodes."1".package_folder' $tmp

Copy-Item -Recurse -Force `
    (Join-Path $libFolder "*") `
    $OutputDir
