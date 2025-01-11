import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy.spatial.transform import Rotation as R
import socket
import sys
from PyQt5 import QtWidgets
import time
import ahrs
from collections import deque

CANT_SAMPLES = 1
BAUD_RATE = 115200
DT = 0.005
DT_2 = 0.005

# Distancia al eje de giro
d = np.array([0.35, 0, 0])
# Gravedad en el sistema global
gravity_global = np.array([0.0, 0.0, 9.8])

# Matrices del sistema
# Estado inicial: posición y velocidad [x, y, z, vx, vy, vz]
x = np.array([[0], [0], [0], [0], [0], [0]])  # Estado inicial: posición y velocidad en 3D
x_estimado = np.zeros(6)

# Matriz de transición de estado (F)
F = np.array([[1, 0, 0, DT_2,  0,  0],
              [0, 1, 0,  0, DT_2,  0],
              [0, 0, 1,  0,  0, DT_2],
              [0, 0, 0,  1,  0,  0],
              [0, 0, 0,  0,  1,  0],
              [0, 0, 0,  0,  0,  1]])

# Matriz de control (B) para incluir aceleración medida
B = np.array([[0.5 * DT_2**2, 0, 0],
              [0, 0.5 * DT_2**2, 0],
              [0, 0, 0.5 * DT_2**2],
              [DT_2, 0, 0],
              [0, DT_2, 0],
              [0, 0, DT_2]])

# Matriz de covarianza inicial (P)
P = np.eye(6) * 0.01  # Incertidumbre inicial en posición y velocidad

# Matriz de covarianza del proceso (Q): incertidumbre del modelo
Q_efk = np.eye(6) * 0.8  # Pequeñas incertidumbres en posición y velocidad

# Matriz de covarianza de las mediciones (R): incertidumbre del sensor
R = np.eye(3) * 0.1  # Ruido del acelerómetro en las tres dimensiones

# Matriz de medición (H): sólo medimos aceleración (relacionada con la velocidad)
H = np.array([[0, 0, 0, 1/DT_2, 0, 0],
              [0, 0, 0, 0, 1/DT_2, 0],
              [0, 0, 0, 0, 0, 1/DT_2]])


# Función para corregir el efecto de la gravedad
def remove_gravity(acc, Q):
    if (np.linalg.norm(acc) < 9.9):
        return np.array([0.0, 0.0, 0.0])
    Q = quaternion_inverse(Q)
    gravity_corr = np.array(rotate_vector_by_quaternion(gravity_global, Q))
    # print("Gravedad corregida: ",gravity_corr)
    # print("Aceleración: ",acc)
    # Aceleración debida al movimiento
    acc_corr = acc - gravity_corr
    # print("Aceleración corregida: ",acc_corr)
    # if abs(acc_corr[0]) < 0.7:
    #     acc_corr[0] = 0
    # if abs(acc_corr[1]) < 0.7:
    #     acc_corr[1] = 0
    # if abs(acc_corr[2]) < 0.7:
    #     acc_corr[2] = 0
    return acc_corr

# Función para rotar puntos usando un cuaternión
def rotate_points(points, quaternion):
    w, x, y, z = quaternion
    # Matriz de rotación derivada del cuaternión
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])
    # Aplicar la rotación
    return np.dot(points, R.T)

class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        
    def filter(self, measurement):
        # Agrega la nueva medición al final de la ventana
        self.window.append(measurement)
        
        # Calcula y devuelve el promedio de la ventana
        return np.mean(self.window, axis=0)

class LowPassFilter:
    def __init__(self, alpha=0.1):
        # Factor de suavizado (0 < alpha < 1)
        self.alpha = alpha
        self.state = None
    
    def filter(self, measurement):
        measurement = np.array(measurement, dtype=float)
        if self.state is None:
            # Inicializar el estado con la primera medición
            self.state = measurement
        else:
            # Aplicar el filtro pasabajos exponencial
            self.state = self.alpha * measurement + (1 - self.alpha) * self.state
        return self.state

# Función de corrección del giroscopio
def compensate_gyroscope(orientation, gyro_measurement, d):
    # Convierte la orientación a matriz de rotación
    rotation = R.from_quat(orientation)
    rotation_matrix = rotation.as_matrix()
    d_rot = rotation.apply(d)

    # Calcular el efecto de traslación: velocidad angular inducida
    omega_translation = np.cross(gyro_measurement, rotation_matrix @ d_rot)
    
    # Restar la velocidad de traslación de las mediciones del giroscopio
    gyro_corrected = gyro_measurement - omega_translation
    
    return gyro_corrected

# Función de corrección del acelerómetro
def compensate_accelerometer(orientation, accel_measurement, gyro_measurement, prev_gyro_measurement, delta_t, d):
    # Convierte la orientación a matriz de rotación
    rotation = R.from_quat(orientation)
    rotation_matrix = rotation.as_matrix()
    d_rot = rotation.apply(d)

    # Aplica la rotación actual al desplazamiento para alinear con la orientación
    rotated_d = rotation_matrix @ d_rot  # Desplazamiento en el sistema de referencia del sensor
    
    # Aceleración centrípeta
    omega_magnitude = np.linalg.norm(gyro_measurement)
    a_centripeta = - omega_magnitude**2 * rotated_d
    
    # Aceleración tangencial
    angular_acceleration = (gyro_measurement - prev_gyro_measurement) / delta_t 
    
    # Aceleración tangencial
    a_tangencial = np.cross(angular_acceleration, rotated_d)
    
    # Compensar las lecturas del acelerómetro
    accel_corrected = accel_measurement - a_centripeta - a_tangencial
    
    return accel_corrected

# Función de corrección del magnetómetro
def compensate_magnetometer(orientation, mag_measurement, d):
    # Transformar el campo magnético al sistema del IMU
    rotation = R.from_quat(orientation)
    rotation_matrix = rotation.as_matrix()
    d_rot = rotation.apply(d)
    local_magnetic_field = -40.0
    
    # Calcular el campo magnético corregido
    mag_corrected = rotation_matrix @ (local_magnetic_field + np.cross(rotation_matrix @ d_rot, mag_measurement))
    
    return mag_corrected

# Función para interpretar los datos recibidos por WiFi
def parse_sensor_data(data_string, ref):
    try:
        data = [float(value) for value in data_string.split(',')]
        if len(data) == (9 * CANT_SAMPLES):
            return data[ref:ref+3], data[ref+3:ref+6], data[ref+6:ref+9]  # Acelerómetro, Giroscopio, Magnetómetro
        else:
            return None
    except:
        return None

# Función para recibir una cantidad específica de bytes
def receive_until_newline(client_socket):
    data = b''
    while not data.endswith(b'\n'):  # Recibir hasta encontrar '\n'
        packet = client_socket.recv(1)  # Recibir un byte a la vez
        if not packet:
            return None  # Si no se reciben más datos, retornar None
        data += packet
    return data

# Función para crear la caja
def create_box(size=(1, 1, 1)):
    l, w, h = size
    vertices = np.array([[0, -w/2, -h/2],
                         [l, -w/2, -h/2],
                         [l, w/2, -h/2],
                         [0, w/2, -h/2],
                         [0, -w/2, h/2],
                         [l, -w/2, h/2],
                         [l, w/2, h/2],
                         [0, w/2, h/2]])
    return vertices

# Configuración inicial del punto
def update_point(x, y, z):
    data = np.array([[x, y, z]])
    point.setData(pos=data, size=10, color=(1, 0, 0, 1))

def quaternion_multiply(q1, q2):
    """
    Multiplica dos cuaterniones.
    q1 y q2 son listas de longitud 4: [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return [w, x, y, z]

def quaternion_conjugate(q):
    """
    Calcula el conjugado de un cuaternión.
    """
    w, x, y, z = q
    return [w, -x, -y, -z]

def rotate_vector_by_quaternion(vector, quaternion):
    """
    Rota un vector 3D utilizando un cuaternión.
    - vector: Lista de 3 elementos [vx, vy, vz].
    - quaternion: Lista de 4 elementos [qw, qx, qy, qz].
    Devuelve el vector rotado como lista de 3 elementos.
    """
    # Convertir el vector a un cuaternión puro (parte escalar = 0)
    vector_quat = [0, 0, 0, 0]
    vector_quat[3] = vector[2]
    vector_quat[2] = vector[1]
    vector_quat[1] = vector[0]

    # Calcular el conjugado del cuaternión
    quat_conjugate = quaternion_conjugate(quaternion)

    # Rotar el vector: q * v * q^-1
    temp_result = quaternion_multiply(quaternion, vector_quat)
    rotated_quat = quaternion_multiply(temp_result, quat_conjugate)

    # Extraer la parte vectorial del cuaternión resultante
    rotated_vector = rotated_quat[1:]  # Ignorar la parte escalar
    return rotated_vector

def quaternion_inverse(q):
    """
    Calcula el cuaternión inverso para un cuaternión dado en formato [w, x, y, z].
    
    Args:
        q: np.array con el cuaternión [w, x, y, z].

    Returns:
        np.array con el cuaternión inverso [w, -x, -y, -z] / |q|^2.
    """
    w, x, y, z = q
    
    # Calcular la norma del cuaternión
    norm = w**2 + x**2 + y**2 + z**2
    
    # Cuaternión conjugado
    q_conjugate = np.array([w, -x, -y, -z])
    
    # Cuaternión inverso (conjugado / norma^2)
    q_inverse = q_conjugate / norm
    
    return q_inverse

# Función de actualización para la animación
def update():
    global acc, gyro, mag, Q, Q_buff, end_time, start_time

    # start_time = time.time()
    # wait_time = start_time - end_time
    # print(f"Tiempo de espera entre update's: {wait_time:.4f} segundos")

    raw_data = receive_until_newline(client_socket)  # Recibe hasta 1024 bytes
    if not raw_data:
        return

    sensor_data = parse_sensor_data(raw_data.decode(), 0)
    if sensor_data is not None:
        acc, gyro, mag = sensor_data

        # Aplicar rotaciones
        # buff = acc[1]
        # acc[1] = acc[0]
        # acc[0] = buff
        acc[1] = -acc[1]
        acc = [x * 9.8 for x in acc]

        #buff = gyro[1]
        # gyro[1] = -gyro[0]
        # gyro[0] = -buff
        gyro[0] = -gyro[0]
        gyro[2]= -gyro[2]
        gyro = np.radians(gyro)  # Convertir giroscopio a rad/s
        
        # buff = mag[1]
        # mag[1] = mag[0]
        # mag[0] = buff
        mag[1] = -mag[1]
        mag[0] = -mag[0]

        #filtrado
        # gyro = filtro_pb.filter(gyro)
        # gyro = filtro_mm.filter(gyro)

        #gyro = compensate_gyroscope(Q_buff, gyro, d)
        #acc = compensate_accelerometer(Q_buff, acc, gyro, gyro_prev, DT, d)
        #mag = compensate_magnetometer(Q_buff, mag, d)

        Q = efk.update(Q_buff, gyro, acc, mag, DT)
        Q_buff = Q

        # Rotar el vector de gravedad
        rotated_end_gravedad = rotate_vector_by_quaternion(end_gravedad, Q)
        rotated_vector_gravedad = np.array([start_gravedad, rotated_end_gravedad])
        vector_grav.setData(pos=rotated_vector_gravedad)

        # Rotar el vector de aceleracion
        new_vector_acc = np.array([start_gravedad, acc])
        vector_acc.setData(pos=new_vector_acc)

        # Rotar los vértices de la caja
        vertices_rotados = rotate_points(vertices_ned, Q)
        # Volver al sistema original (invertir el eje Z, intercambiar X e Y)
        vertices_rotados_buff = vertices_rotados.copy()
        vertices_rotados_buff[:, 2] = -vertices_rotados[:, 2]
        vertices_rotados_buff[:, 1] = -vertices_rotados[:, 1]
        # Actualizar la geometría de la caja en el gráfico
        meshdata.setVertexes(vertices_rotados_buff)
        box.meshDataChanged()

    # end_time = time.time()
    # cicle_time = end_time - start_time
    # print(f"Tiempo de procesamiento del ciclo: {cicle_time:.4f} segundos")

def pos():
    global acc, Q, F, x, B, P, H, R, Q_efk, x_estimado, end_time, start_time

    #filtrado
    # acc = filtro_pb.filter(acc)
    # acc = filtro_mm.filter(acc)

    # start_time = time.time()
    # wait_time = start_time - end_time
    # print(f"Tiempo de espera entre update's: {wait_time:.4f} segundos")
    
    # Transforma aceleración al sistema global
    acc_global = remove_gravity(acc, Q)

    # 1. Predicción
    u = acc_global.reshape(3, 1)  # Entrada de control (aceleración medida)
    x = np.dot(F, x) + np.dot(B, u)  # Predicción del estado
    P = np.dot(np.dot(F, P), F.T) + Q_efk  # Predicción de la covarianza

    # 2. Actualización
    # Innovación (residuo): diferencia entre medición y predicción
    y = acc_global.reshape(3, 1) - np.dot(H, x)
    
    # Ganancia de Kalman
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))

    # Corrección del estado
    x = x + np.dot(K, y)

    # Actualización de la covarianza
    P = P - np.dot(np.dot(K, H), P)

    # Simplificar el estado estimado
    x_estimado = x.flatten()

    # Convertir las estimaciones en arrays para graficar
    x_estimado = np.array(x_estimado)

    # Imprimir resultados
    update_point(x_estimado[0], x_estimado[1], -x_estimado[2])

    # print("Aceleracion en X: ", acc_global[0])
    # print("Aceleracion en Y: ", acc_global[1])
    # print("Aceleracion en Z: ", acc_global[2])
    print("Posición en X: ", x_estimado[0]*100)
    print("Posición en Y: ", x_estimado[1]*100)
    print("Posición en Z: ", -x_estimado[2]*100)

    # end_time = time.time()
    # cicle_time = end_time - start_time
    # print(f"Tiempo de procesamiento del ciclo: {cicle_time:.4f} segundos")

# Dirección IP y puerto del ESP32
esp32_ip = '192.168.0.186'  # Reemplázalo con la IP de tu ESP32
esp32_port = 80             # El mismo puerto que configuraste en el ESP32

# Crear un socket TCP/IP
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Variables para almacenar los datos del sensor
acc = np.zeros(3)  # Acelerómetro (X, Y, Z)
acc_new = np.zeros(3)  # Acelerómetro (X, Y, Z)
gyro = np.zeros(3)  # Giroscopio (X, Y, Z)
acc_prev = np.zeros(3)  # Acelerómetro previo (X, Y, Z)
mag = np.zeros(3)   # Magnetómetro (X, Y, Z)

#Filtros
filtro_pb = LowPassFilter(alpha=0.1)
filtro_mm = MovingAverageFilter(window_size=5)


# Crear una aplicación de PyQt
app = QtWidgets.QApplication([])

# Crear una ventana OpenGL
view = gl.GLViewWidget()
view.show()
view.setWindowTitle('3D Box Real-Time Visualization')
view.setCameraPosition(distance=6, azimuth=90)

# Agregar una rejilla para referencia
grid = gl.GLGridItem()
grid.setSize(x=2, y=2, z=2)
grid.setSpacing(x=1, y=1, z=1)
view.addItem(grid)

# Crear un sistema de ejes
axes = gl.GLAxisItem()
axes.setSize(4, 4, 4)  # Cambia el tamaño según tus necesidades
view.addItem(axes)

# Definir el vector
start_gravedad = np.array([0, 0, 0])  # Origen del vector
end_gravedad = np.array([0, 0, 1])    # Extremo del vector
vector_gravedad = np.array([start_gravedad, end_gravedad])

# Crear el vector como una línea
vector_grav = gl.GLLinePlotItem(
    pos=vector_gravedad,
    color=(1, 1, 1, 1),  # Color blanco
    width=4,             # Grosor de la línea
    mode='lines'
)
view.addItem(vector_grav)

# Definir el vector
start_acc = np.array([0, 0, 0])  # Origen del vector
end_acc = np.array([0, 0, 1])    # Extremo del vector
vector_acc = np.array([start_acc, end_acc])

# Crear el vector como una línea
vector_acc = gl.GLLinePlotItem(
    pos=vector_acc,
    color=(1, 0.5, 0, 1),  # Color naranja
    width=4,             # Grosor de la línea
    mode='lines'
)
view.addItem(vector_acc)

# Tamaño de la caja
box_size = (2, 0.05, 0.05)
vertices  = create_box(size=box_size)
# # Transformar al sistema NED (invertir el eje Z, intercambiar X e Y)
vertices_ned = vertices.copy()
vertices_ned[:, 2] = -vertices[:, 2]
vertices_ned[:, 1] = -vertices[:, 1]

# Crear índices para dibujar caras de la caja
faces = np.array([
    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4]
])

# Crear el objeto Mesh
meshdata = gl.MeshData(vertexes=vertices, faces=faces)
box = gl.GLMeshItem(meshdata=meshdata, color=(0, 1, 1, 0.5), smooth=False, drawEdges=True, edgeColor=(1, 0, 0, 1))
view.addItem(box)

# Configuración inicial de PyQtGraph
app_2 = QtWidgets.QApplication(sys.argv)
window_2 = gl.GLViewWidget()
window_2.show()
window_2.setWindowTitle('Gráfico 3D en Tiempo Real')
window_2.setCameraPosition(distance=0.5)

# Crear los ejes
axis_2 = gl.GLAxisItem()
axis_2.setSize(10, 10, 10)
window_2.addItem(axis_2)

# Crear el punto en 3D
point = gl.GLScatterPlotItem()
window_2.addItem(point)

Q = np.zeros((1, 4))  # Allocate array for quaternions
Q_buff = np.array([0.7071, 0, 0.7071, 0])

efk = ahrs.filters.EKF(frequency=(1/DT), magnetic_ref=-40.0, noises=[0.4**2, 0.05**2, 0.2**2], P=(np.identity(4)*100), q0=Q_buff)
start_time = 0
end_time = 0

# Conectar al ESP32
client_socket.connect((esp32_ip, esp32_port))
print(f"Conectado al ESP32 en {esp32_ip}:{esp32_port}")




# Temporizador para actualizar la gráfica en tiempo real
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)  # Actualiza cada 2.5 ms
# Temporizador para actualizar la gráfica en tiempo real
timer_2 = pg.QtCore.QTimer()
timer_2.timeout.connect(pos)
timer_2.start(3)  # Actualiza cada 2.5 ms

# Iniciar la aplicación de PyQt
sys.exit(app.exec_())
sys.exit(app_2.exec_())