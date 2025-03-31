import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para proyección 3D
from matplotlib.animation import FuncAnimation

def moving_average(data, window_size=5):
    """
    Aplica un filtro de promedio móvil (ventana de tamaño window_size)
    para suavizar los datos.
    """
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Leer datos desde el CSV con columnas timestamp;X;Y;Z
datos = []
with open('posicion_onlycam_V2.csv', 'r', newline='') as csvfile:
    lector = csv.DictReader(csvfile, delimiter=';')
    for fila in lector:
        timestamp = float(fila['timestamp'])
        x = float(fila['X'])
        y = float(fila['Y'])
        z = float(fila['Z'])
        datos.append((timestamp, x, y, z))
datos = np.array(datos)

# Aplicar el filtro pasabajos (promedio móvil) a cada coordenada
window_size = 5  # Ajusta este parámetro según el nivel de suavizado deseado
timestamps = datos[:, 0]
x_filtered = moving_average(datos[:, 1], window_size)
y_filtered = moving_average(datos[:, 2], window_size)
z_filtered = moving_average(datos[:, 3], window_size)

# Combinar los datos filtrados
datos_filtrados = np.column_stack((timestamps, x_filtered, y_filtered, z_filtered))

# Configuración de la figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Determinar límites basados en los datos filtrados (puedes ajustarlos si lo deseas)
ax.set_xlim(np.min(x_filtered)-1, np.max(x_filtered)+1)
ax.set_ylim(np.min(y_filtered)-1, np.max(y_filtered)+1)
ax.set_zlim(np.min(z_filtered)-1, np.max(z_filtered)+1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# -------------------------------------------------------------------
# Definir el recorrido preestablecido con 5 puntos de referencia
# Estos valores son ejemplos; puedes modificarlos según tu ruta deseada.
reference_points = np.array([
    [0, 0, 0],
    [-0.62, 0, 0.22],
    [-0.73, 0, -0.43],
    [0.03, 0.3, -0.30],
    [0, 0, 0]
])

# Dibujar la ruta preestablecida como línea discontinua y marcar los puntos
ax.plot(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2],
        'g--', linewidth=2, label='Ruta preestablecida')
ax.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2],
           color='green', s=100, label='Puntos de referencia')
# -------------------------------------------------------------------

# Variables para almacenar la trayectoria del punto animado
xs, ys, zs = [], [], []
point = None
line, = ax.plot([], [], [], color='blue', linewidth=2)  # Línea de rastro (trayectoria)

def update(frame):
    global xs, ys, zs, point, line
    timestamp, x, y, z = datos_filtrados[frame]
    xs.append(x)
    ys.append(y)
    zs.append(z)
    
    # Remover el punto anterior si existe
    if point is not None:
        point.remove()
    
    # Dibujar el punto actual en rojo
    point = ax.scatter(x, y, z, color='red', s=50)
    
    # Actualizar la línea del rastro con los puntos acumulados
    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    
    ax.set_title(f'Tiempo: {timestamp:.2f}')

# Crear la animación: se recorre la trayectoria una sola vez (repeat=False)
ani = FuncAnimation(fig, update, frames=len(datos_filtrados), interval=0, repeat=False)

plt.legend()
plt.show()