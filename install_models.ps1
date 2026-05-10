# ODC Markets Research Agent - Model Installer
# Run this script once to pull all required Ollama models.
# Right-click -> "Run with PowerShell", or run from terminal:
#   powershell -ExecutionPolicy Bypass -File install_models.ps1

$models = @(
    @{ name = "qwen2.5:7b";       role = "Planner" },
    @{ name = "qwen2.5-coder:7b"; role = "Strategy Generator" },
    @{ name = "deepseek-r1:8b";   role = "Critic" },
    @{ name = "nomic-embed-text"; role = "Embeddings (Memory)" }
)

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  ODC Markets Agent - Model Installer  " -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# Check Ollama is running
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
    Write-Host "Ollama is running." -ForegroundColor Green
    $installed = $response.models | ForEach-Object { $_.name }
} catch {
    Write-Host "ERROR: Ollama is not running or not installed." -ForegroundColor Red
    Write-Host "  -> Download from https://ollama.com and start it, then re-run this script."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
$total = $models.Count
$i = 0

foreach ($model in $models) {
    $i++
    $name = $model.name
    $role = $model.role

    # Check if already installed
    $alreadyInstalled = $installed | Where-Object { $_ -like "$name*" }
    if ($alreadyInstalled) {
        Write-Host "[$i/$total] $name ($role) - already installed, skipping." -ForegroundColor Yellow
        continue
    }

    Write-Host ""
    Write-Host "[$i/$total] Pulling $name ($role)..." -ForegroundColor Cyan
    $result = ollama pull $name
    if ($LASTEXITCODE -eq 0) {
        Write-Host "         Done." -ForegroundColor Green
    } else {
        Write-Host "         FAILED - check your connection and try again." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  Installed models:" -ForegroundColor Cyan
ollama list
Write-Host ""
Write-Host "  All done! You can now launch the app:" -ForegroundColor Green
Write-Host "  streamlit run app.py" -ForegroundColor White
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close"
