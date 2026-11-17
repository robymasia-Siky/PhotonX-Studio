@echo off
title PhotonX VST3 Installer
cd /d "%~dp0"
echo ========================================================
echo  PhotonX Studio - VST3 Plugin Installer
echo ========================================================

set TARGET_USER=%LOCALAPPDATA%\Programs\Common\VST3\PhotonX.vst3
echo [1/3] Installazione in: %TARGET_USER%
if not exist "%LOCALAPPDATA%\Programs\Common\VST3" mkdir "%LOCALAPPDATA%\Programs\Common\VST3"
xcopy /E /I /Y "build\VST3\Release\PhotonX.vst3" "%TARGET_USER%"

echo [2/3] Copia modelli GGML in C:\ProgramData\OpenPhoton\models...
if not exist "C:\ProgramData\OpenPhoton\models" mkdir "C:\ProgramData\OpenPhoton\models"
copy /Y "models\*.bin" "C:\ProgramData\OpenPhoton\models\"

echo [3/3] Copia modelli GGML all'interno del bundle VST3...
if not exist "%TARGET_USER%\Contents\Resources\models" mkdir "%TARGET_USER%\Contents\Resources\models"
copy /Y "models\*.bin" "%TARGET_USER%\Contents\Resources\models\"

echo.
echo ========================================================
echo  PhotonX.vst3 installato con successo!
echo  Riavvia la tua DAW (Ableton, FL Studio, Reaper, ecc.)
echo  ed esegui una ri-scansione dei plugin VST3.
echo ========================================================
pause
