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

# Crear una única figura con dos subplots 3D (lado a lado)
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

# Función para dibujar la geometría estática (tambores, platillos) en un eje dado.
def plot_instruments(ax):
    # Dibujo de los tambores y Hi-hat.
    radio = 15
    theta = np.linspace(0, 2 * np.pi, 100)
    despl_x = [-9, -11, 32, -41]        # Snare / High-tom / Low-tom / Hi-hat
    despl_z = [-11, -52, -12, -9]        # Snare / High-tom / Low-tom / Hi-hat
    alturas = [60, 17, 50, 30]           # Snare / High-tom / Low-tom / Hi-hat
    for y, z, x in zip(alturas, despl_z, despl_x):
        x_circ = radio * np.cos(theta) + x
        z_circ = radio * np.sin(theta) + z
        # Usamos np.full_like para crear un vector con la altura constante
        ax.plot(x_circ, np.full_like(x_circ, y), z_circ, color='cyan', linestyle='--', alpha=0.7)
    
    # Dibujo de los platillos (Crash / Ride).
    radio = 20
    theta = np.linspace(0, 2 * np.pi, 100)
    despl_x = [-33, 31]         # Crash / Ride
    despl_z = [-36, -41]        # Crash / Ride
    alturas = [17, 20]         # Crash / Ride
    for y, z, x in zip(alturas, despl_z, despl_x):
        x_circ = radio * np.cos(theta) + x
        z_circ = radio * np.sin(theta) + z
        ax.plot(x_circ, np.full_like(x_circ, y), z_circ, color='cyan', linestyle='--', alpha=0.7)
    
    # Configuración de los límites y etiquetas.
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_zlim(-80, 80)
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')

# Dibujar la misma información estática en ambos subplots.
plot_instruments(ax1)
plot_instruments(ax2)

# Crear los elementos gráficos (línea de trayectoria y punto actual) para cada subplot.
trail_line1, = ax1.plot([], [], [], 'r-', lw=2, label='Trayectoria')
current_point1, = ax1.plot([], [], [], 'bo', markersize=8, label='Punto actual')
trail_line2, = ax2.plot([], [], [], 'r-', lw=2, label='Trayectoria')
current_point2, = ax2.plot([], [], [], 'bo', markersize=8, label='Punto actual')

# Lista para almacenar los puntos de la trayectoria.
trail_points = []

# Función de actualización de la animación.
def actualizar(frame):
    # Leemos las coordenadas actuales desde la memoria compartida y escalamos a cm.
    x = coords[0] * 100
    y = coords[1] * 100
    z = coords[2] * 100
    tiempo_actual = time.time()
    trail_points.append((tiempo_actual, (x, y, z)))
    
    # Eliminamos los puntos con más de 2 segundos de antigüedad.
    while trail_points and (tiempo_actual - trail_points[0][0] > 1):
        trail_points.pop(0)
    
    # Actualizar la trayectoria si hay puntos.
    if trail_points:
        data = np.array([p for _, p in trail_points])
        # Actualización para el primer subplot
        trail_line1.set_data(data[:, 0], data[:, 1])
        trail_line1.set_3d_properties(data[:, 2])
        # Actualización para el segundo subplot
        trail_line2.set_data(data[:, 0], data[:, 1])
        trail_line2.set_3d_properties(data[:, 2])
    else:
        trail_line1.set_data([], [])
        trail_line1.set_3d_properties([])
        trail_line2.set_data([], [])
        trail_line2.set_3d_properties([])
    
    # Actualizar el punto actual en ambos subplots.
    current_point1.set_data([x], [y])
    current_point1.set_3d_properties([z])
    current_point2.set_data([x], [y])
    current_point2.set_3d_properties([z])
    
    return trail_line1, current_point1, trail_line2, current_point2

# Crear la animación utilizando la figura que contiene ambos subplots.
ani = FuncAnimation(fig, actualizar, interval=100, blit=False)

# Agregar leyendas a cada eje.
ax1.legend()
ax2.legend()

plt.show()

# Al cerrar la ventana, liberamos la conexión a la memoria compartida.
shm.close()