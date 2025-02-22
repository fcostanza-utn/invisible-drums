import threading
import time

class CustomTimer:
    def __init__(self, interval, function):
        self.interval = interval  # Intervalo en segundos
        self.function = function  # Función a ejecutar
        self._running = False
        self._thread = None

    def _run(self):
        while self._running:
            time.sleep(self.interval)
            if self._running:  # Verifica si no se detuvo durante el sueño
                self.function()

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
"""
# Ejemplo de uso
def my_function():
    print("¡Temporizador activado!")

timer = CustomTimer(2, my_function)  # Temporizador que ejecuta cada 2 segundos
timer.start()

# Detener el temporizador después de 10 segundos
time.sleep(10)
timer.stop()
"""

