@echo off
title PhotonX Studio - Model Downloader
cd /d "%~dp0"
echo ========================================================
echo  PhotonX Studio - Pre-trained AI Models Downloader
echo ========================================================
python tools\download_models.py
pause
