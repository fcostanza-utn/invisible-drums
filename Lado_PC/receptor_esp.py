import numpy as np
from multiprocessing import shared_memory
from esp_module import IMUVisualizer
import sys
import time
from RW_LOCK import RWLock

"""
INICIALIZACION DE IMU
"""
visualizer = IMUVisualizer('192.168.1.70', 80, 0.02)
rwlock = RWLock()

# Nombre del bloque de memoria compartida (debe coincidir con el que usa el productor)
SHM_NAME = 'esp_shared'

# Se crea la memoria compartida para 3 valores de tipo double.
# El tamaño se calcula como 3 * tamaño de un double.
shm_esp = shared_memory.SharedMemory(create=True, name=SHM_NAME, size=1 * np.dtype('U100').itemsize)
# Creamos un array numpy que utiliza el buffer de la memoria compartida.
imu_data = np.ndarray((1,), dtype='U100', buffer=shm_esp.buf)

# Mantenemos el hilo principal vivo (por ejemplo, con un bucle infinito o esperando a que terminen los hilos)
try:
    while True:
        rwlock.acquire_write()
        try:
            data = visualizer.receive_data()  # Leer IMU
            if data:
                imu_data[...] = data
                print("data: ", data)
        finally:
            rwlock.release_write()
        time.sleep(0.005)
except KeyboardInterrupt:
    print("Terminando la ejecución...")
    shm_esp.close()
    shm_esp.unlink()
    sys.exit()