@echo off

start "" py receptor_esp.py
timeout /t 15 /nobreak

start "" py main.py
timeout /t 10 /nobreak

start "" py graficador.py
pause