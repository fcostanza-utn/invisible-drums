import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from multiprocessing import shared_memory

# Nombre que se usará para identificar la memoria compartida del graficador.
SHM_NAME_GRAF_RIGHT = 'coords_shared_right'
SHM_NAME_GRAF_LEFT = 'coords_shared_left'

# Intentamos conectar con la memoria compartida ya creada
try:
    shm_right = shared_memory.SharedMemory(name=SHM_NAME_GRAF_RIGHT)
except FileNotFoundError:
    print(f"No se encontró la memoria compartida con nombre '{SHM_NAME_GRAF_RIGHT}'. Asegúrate de que el productor la haya creado.")
    exit(1)

# Intentamos conectar con la memoria compartida ya creada
try:
    shm_left = shared_memory.SharedMemory(name=SHM_NAME_GRAF_LEFT)
except FileNotFoundError:
    print(f"No se encontró la memoria compartida con nombre '{SHM_NAME_GRAF_LEFT}'. Asegúrate de que el productor la haya creado.")
    exit(1)

# Creamos un array numpy que utiliza el buffer de la memoria compartida.
# Se asume que el productor escribe 3 valores (x, y, z) de tipo double.
coords_right = np.ndarray((3,), dtype='d', buffer=shm_right.buf)
coords_left = np.ndarray((3,), dtype='d', buffer=shm_left.buf)

# Crear una única figura con dos subplots 3D (lado a lado)
fig, ax = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

ax1 = ax[0, 0]
ax2 = ax[0, 1]
ax3 = ax[1, 0]
ax4 = ax[1, 1]

# Función para dibujar la geometría estática (tambores, platillos) en un eje dado.
def plot_instruments(ax):
    # Dibujo de los tambores y Hi-hat.
    radio = 15
    theta = np.linspace(0, 2 * np.pi, 100)
    despl_x = [-9, -11, 32, -41]        # Snare / High-tom / Low-tom / Hi-hat
    despl_z = [-11, -52, -12, -9]        # Snare / High-tom / Low-tom / Hi-hat
    alturas = [50, 17, 50, 30]           # Snare / High-tom / Low-tom / Hi-hat
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
plot_instruments(ax3)
plot_instruments(ax4)

# Crear los elementos gráficos (línea de trayectoria y punto actual) para cada subplot.
trail_line1, = ax1.plot([], [], [], 'r-', lw=2, label='Trayectoria')
current_point1, = ax1.plot([], [], [], 'bo', markersize=8, label='Punto actual')
trail_line2, = ax2.plot([], [], [], 'r-', lw=2, label='Trayectoria')
current_point2, = ax2.plot([], [], [], 'bo', markersize=8, label='Punto actual')
trail_line3, = ax3.plot([], [], [], 'r-', lw=2, label='Trayectoria')
current_point3, = ax3.plot([], [], [], 'bo', markersize=8, label='Punto actual')
trail_line4, = ax4.plot([], [], [], 'r-', lw=2, label='Trayectoria')
current_point4, = ax4.plot([], [], [], 'bo', markersize=8, label='Punto actual')

# Lista para almacenar los puntos de la trayectoria.
trail_points_right = []
trail_points_left = []


# Función de actualización de la animación.
def actualizar(frame):
    # Leemos las coordenadas actuales desde la memoria compartida y escalamos a cm.
    x_right = coords_right[0] * 100
    y_right = coords_right[1] * 100
    z_right = coords_right[2] * 100
    x_left = coords_left[0] * 100
    y_left = coords_left[1] * 100
    z_left = coords_left[2] * 100
    tiempo_actual = time.time()
    trail_points_right.append((tiempo_actual, (x_right, y_right, z_right)))
    trail_points_left.append((tiempo_actual, (x_left, y_left, z_left)))
    
    # Eliminamos los puntos con más de 2 segundos de antigüedad.
    while trail_points_right and (tiempo_actual - trail_points_right[0][0] > 1):
        trail_points_right.pop(0)
    while trail_points_left and (tiempo_actual - trail_points_left[0][0] > 1):
        trail_points_left.pop(0)
    
    # Actualizar la trayectoria si hay puntos.
    if trail_points_right:
        data = np.array([p for _, p in trail_points_right])
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
    if trail_points_left:
        data = np.array([p for _, p in trail_points_left])
        # Actualización para el primer subplot
        trail_line3.set_data(data[:, 0], data[:, 1])
        trail_line3.set_3d_properties(data[:, 2])
        # Actualización para el segundo subplot
        trail_line4.set_data(data[:, 0], data[:, 1])
        trail_line4.set_3d_properties(data[:, 2])
    else:
        trail_line3.set_data([], [])
        trail_line3.set_3d_properties([])
        trail_line4.set_data([], [])
        trail_line4.set_3d_properties([])
    
    # Actualizar el punto actual en ambos subplots.
    current_point1.set_data([x_right], [y_right])
    current_point1.set_3d_properties([z_right])
    current_point2.set_data([x_right], [y_right])
    current_point2.set_3d_properties([z_right])
    current_point3.set_data([x_left], [y_left])
    current_point3.set_3d_properties([z_left])
    current_point4.set_data([x_left], [y_left])
    current_point4.set_3d_properties([z_left])
    
    return trail_line1, current_point1, trail_line2, current_point2, trail_line3, current_point3, trail_line4, current_point4

# Crear la animación utilizando la figura que contiene ambos subplots.
ani = FuncAnimation(fig, actualizar, interval=100, blit=False)

# Agregar leyendas a cada eje.
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()

# Al cerrar la ventana, liberamos la conexión a la memoria compartida.
shm_right.close()
shm_left.close()