@echo off

start "" py receptor_esp.py
timeout /t 5 /nobreak

start "" py main.py
timeout /t 5 /nobreak

py graficador.py
pause