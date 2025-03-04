import numpy as np
from multiprocessing import shared_memory
import sys
import time
import socket
from RW_LOCK import RWLock

"""
INICIALIZACION DE IMU
"""
rwlock = RWLock()

esp32_ip = '192.168.1.70'
esp32_port = 80

"""
INICIALIZACION SHARED MEMORY
"""
# Nombre del bloque de memoria compartida (debe coincidir con el que usa el productor)
SHM_NAME = 'esp_shared'

# Se crea la memoria compartida para 3 valores de tipo double.
# El tamaño se calcula como 3 * tamaño de un double.
shm_esp = shared_memory.SharedMemory(create=True, name=SHM_NAME, size=1 * np.dtype('U120').itemsize)
# Creamos un array numpy que utiliza el buffer de la memoria compartida.
imu_data = np.ndarray((1,), dtype='U120', buffer=shm_esp.buf)

"""
FUNCIONES
"""
def receive_data():
    data = b''
    while not data.endswith(b'\n'):
        packet = client_socket.recv(1)
        if not packet:
            return None
        data += packet
    return data.decode()

def connect_to_esp32():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((esp32_ip, esp32_port))
            s.settimeout(5)  # Timeout de 5 segundos
            print("Conectado al ESP32")
            return s
        except socket.error as e:
            print("Error de conexión:", e)
            time.sleep(5)  # Espera antes de reintentar

"""
MAIN
"""
client_socket = connect_to_esp32()

try:
    while True:
        rwlock.acquire_write()
        try:
            data = receive_data()  # Leer datos de la conexión
            if data is None:
                print("Conexión perdida, reconectando...")
                client_socket.close()
                client_socket = connect_to_esp32()
                continue  # Vuelve al inicio del bucle tras reconectar
            imu_data[...] = data
            print("imu_data: ", imu_data)
        except socket.timeout:
            print("Timeout en la recepción, reconectando...")
            client_socket.close()
            client_socket = connect_to_esp32()
        except socket.error as e:
            print("Error de socket:", e)
            client_socket.close()
            client_socket = connect_to_esp32()
        finally:
            rwlock.release_write()
        time.sleep(0.005)
except KeyboardInterrupt:
    print("Terminando la ejecución...")
    client_socket.close()
    shm_esp.close()
    shm_esp.unlink()
    sys.exit()
