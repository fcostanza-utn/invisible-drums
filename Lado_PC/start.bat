@echo off
start "" py MIDIdrum.py
timeout /t 5 /nobreak

start "" py receptor_esp.py
timeout /t 10 /nobreak

start "" py main.py
timeout /t 10 /nobreak

start "" py graficador.py
pause