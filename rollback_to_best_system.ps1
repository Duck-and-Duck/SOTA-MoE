Write-Host "=========================================================================" -ForegroundColor Red
Write-Host "          SISTEM EN IYI CALISAN ORİJİNAL DURUMA GERI YUKLENIYOR          " -ForegroundColor Red
Write-Host "=========================================================================" -ForegroundColor Red

$vaultDir = "_SAFE_VAULT_"
if (Test-Path $vaultDir) {
    Get-ChildItem -Path $vaultDir | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination "." -Force
        Write-Host "  [GERI YUKLENDI]: $($_.Name)" -ForegroundColor Green
    }
    Write-Host "`n>>> ISLEM BASARILI: Sistem NASA MAST verisinde 4/4 yapan orijinal duruma donduruldu." -ForegroundColor Cyan
} else {
    Write-Host "[HATA]: _SAFE_VAULT_ klasoru bulunamadi!" -ForegroundColor Red
}
