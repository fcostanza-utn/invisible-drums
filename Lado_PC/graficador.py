import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from multiprocessing import shared_memory

# Nombre del bloque de memoria compartida (debe coincidir con el que usa el productor)
SHM_NAME = 'coords_shared'

# Intentamos conectar con la memoria compartida ya creada
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
except FileNotFoundError:
    print(f"No se encontró la memoria compartida con nombre '{SHM_NAME}'. Asegúrate de que el productor la haya creado.")
    exit(1)

# Creamos un array numpy que utiliza el buffer de la memoria compartida.
# Se asume que el productor escribe 3 valores (x, y, z) de tipo double.
coords = np.ndarray((3,), dtype='d', buffer=shm.buf)

# Configuración del gráfico 3D.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dibujar círculos fijos paralelos al eje Z (planos horizontales a diferentes alturas).
radio = 15  # Radio de 15 cm para un diámetro de 30 cm.
theta = np.linspace(0, 2 * np.pi, 100)
alturas = [0, 10, 20]  # Ejemplo de alturas fijas.
for z in alturas:
    x_circ = radio * np.cos(theta)
    y_circ = radio * np.sin(theta)
    ax.plot(x_circ, y_circ, zs=z, color='gray', linestyle='--', alpha=0.7)

# Configuración de los límites y etiquetas del gráfico.
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(0, 50)
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')

# Elementos gráficos:
# - Una línea para el trazo de la trayectoria.
# - Un punto que indica la posición actual.
trail_line, = ax.plot([], [], [], 'r-', lw=2, label='Trayectoria')
current_point, = ax.plot([], [], [], 'bo', markersize=8, label='Punto actual')

# Lista para almacenar los puntos del trazo (cada elemento es una tupla: (tiempo, (x, y, z))).
trail_points = []

# Función de actualización del gráfico (se ejecuta cada 100ms).
def actualizar(frame):
    # Leemos las coordenadas actuales desde la memoria compartida.
    x = coords[0]
    y = coords[1]
    z = coords[2]
    tiempo_actual = time.time()
    trail_points.append((tiempo_actual, (x, y, z)))
    
    # Se eliminan del rastro los puntos con más de 30 segundos de antigüedad.
    while trail_points and (tiempo_actual - trail_points[0][0] > 30):
        trail_points.pop(0)
    
    # Actualizar la línea de la trayectoria.
    if trail_points:
        data = np.array([p for _, p in trail_points])
        trail_line.set_data(data[:, 0], data[:, 1])
        trail_line.set_3d_properties(data[:, 2])
    else:
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
    
    # Actualizar la posición del punto actual.
    current_point.set_data([x], [y])
    current_point.set_3d_properties([z])
    
    return trail_line, current_point

# Creamos la animación que se actualiza cada 100ms.
ani = FuncAnimation(fig, actualizar, interval=100, blit=False)

plt.legend()
plt.show()

# Al cerrar la ventana, liberamos la conexión a la memoria compartida.
shm.close()