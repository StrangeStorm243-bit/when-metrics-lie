# Structure sanity check for web/frontend
# Ensures no accidental nested web/frontend/web directory exists

$frontendRoot = $PSScriptRoot + "\.."
$nestedPath = Join-Path $frontendRoot "web"

Write-Host "Checking frontend structure..."
Write-Host "Frontend root: $frontendRoot"
Write-Host ""

if (Test-Path $nestedPath) {
    Write-Host "WARNING: Nested directory found: $nestedPath" -ForegroundColor Yellow
    Write-Host ""
    
    $files = Get-ChildItem -Path $nestedPath -Recurse -File -ErrorAction SilentlyContinue
    
    if ($files) {
        Write-Host "ERROR: Found $($files.Count) file(s) in nested directory:" -ForegroundColor Red
        foreach ($file in $files) {
            Write-Host "  - $($file.FullName)" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "This is dangerous! Files should be in canonical locations under web/frontend/" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "Directory exists but is empty (only subdirectories)." -ForegroundColor Yellow
        Write-Host "Safe to delete when not locked by IDE/file explorer." -ForegroundColor Yellow
        Write-Host ""
        exit 0
    }
} else {
    Write-Host "OK: No nested directory found. Structure is correct." -ForegroundColor Green
    exit 0
}

