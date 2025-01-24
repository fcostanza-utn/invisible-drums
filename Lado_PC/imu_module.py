import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy.spatial.transform import Rotation as R
import socket
import sys
from PyQt5 import QtWidgets
from collections import deque
from ahrs.filters import EKF

class IMUVisualizer:
    class MovingAverageFilter:
        def __init__(self, window_size=5):
            self.window_size = window_size
            self.window = deque(maxlen=window_size)
            
        def filter(self, measurement):
            self.window.append(measurement)
            return np.mean(self.window, axis=0)

    class LowPassFilter:
        def __init__(self, alpha=0.1):
            self.alpha = alpha
            self.state = None
        def filter(self, measurement):
            measurement = np.array(measurement, dtype=float)
            if self.state is None:
                self.state = measurement
            else:
                self.state = self.alpha * measurement + (1 - self.alpha) * self.state
            return self.state

    def __init__(self, esp32_ip, esp32_port, dt=0.005):
        self.CANT_SAMPLES = 1
        self.dt = dt
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port

        # Distancia al eje de giro
        self.d = np.array([0.35, 0, 0])
        # Referencia de acelerómetro
        self.acc_ref = np.array([0, 0, 1])
        # Gravedad en el sistema global
        self.gravity_global = np.array([0.0, 0.0, 9.8])

        # Matrices del sistema
        # Estado inicial: posición y velocidad [x, y, z, vx, vy, vz]
        self.x = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0]])  # Estado inicial: posición, velocidad y aceleracion en 3D
        self.x_estimado = np.zeros(9)

        # Matriz de transición de estado (F)
        self.F = np.array([[1, 0, 0, self.dt, 0,    0,    0.5 * self.dt**2, 0,                         0],
                    [0, 1, 0, 0,    self.dt, 0,    0,             0.5 * self.dt**2, 0            ],
                    [0, 0, 1, 0,    0,    self.dt, 0,             0,             0.5 * self.dt**2],
                    [0, 0, 0, 1,    0,    0,    self.dt,          0,             0            ],
                    [0, 0, 0, 0,    1,    0,    0,             self.dt,          0            ],
                    [0, 0, 0, 0,    0,    1,    0,             0,             self.dt         ],
                    [0, 0, 0, 0,    0,    0,    1,             0,             0            ],
                    [0, 0, 0, 0,    0,    0,    0,             1,             0            ],
                    [0, 0, 0, 0,    0,    0,    0,             0,             1            ]])


        # Matriz de covarianza inicial (P)
        self.P = np.eye(9) * 10  # Incertidumbre inicial en posición y velocidad

        # Matriz de covarianza del proceso (Q): incertidumbre del modelo
        self.Q_efk = np.eye(9) * 0.8  # Pequeñas incertidumbres en posición y velocidad

        # Matriz de covarianza de las mediciones (R): incertidumbre del sensor
        self.R = np.eye(3) * 0.1  # Ruido del acelerómetro en las tres dimensiones

        # Matriz de medición (H):
        self.H = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # Matriz de medición (H_T):
        self.H_t = np.array([[2/self.dt**2, 0, 0, -2/self.dt, 0, 0],
                    [0, 2/self.dt**2, 0, 0, -2/self.dt,0],
                    [0, 0, 2/self.dt**2, 0, 0, -2/self.dt]])

        # Conexión con ESP32
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.esp32_ip, self.esp32_port))

        # Inicialización de sensores
        self.acc = np.zeros(3)
        self.acc_prev = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)
        self.Q = np.zeros((1, 4))  # Allocate array for quaternions
        self.Q_buff = np.array([0.7071, 0, 0.7071, 0])

        #Filtros
        self.filtro_pb = self.LowPassFilter(alpha=0.1)
        self.filtro_mm = self.MovingAverageFilter(window_size=5)

        # Configuración del filtro EKF
        self.ekf = EKF(frequency=(1/dt), magnetic_ref=-40.0, noises=[0.4**2, 0.05**2, 0.2**2], P=(np.identity(4)*100), q0=self.Q_buff)

        self.start_gravedad = np.array([0, 0, 0])
        self.end_gravedad = np.array([0, 0, 1])
        self.start_acc = np.array([0, 0, 0])  # Origen del vector
        self.end_acc = np.array([0, 0, 1])    # Extremo del vector
        self.vector_acc = np.array([self.start_acc, self.end_acc])
        '''
        Graficos QT
        '''
        # Crear ventana 3D
        self.app = QtWidgets.QApplication([])
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('3D Box Real-Time Visualization')
        self.view.setCameraPosition(distance=6, azimuth=90)

        # Configurar objetos visuales
        self._setup_visual_objects_app1()

        self.app_2 = QtWidgets.QApplication(sys.argv)
        self.window_2 = gl.GLViewWidget()
        self.window_2.setWindowTitle('Gráfico 3D en Tiempo Real')
        self.window_2.setCameraPosition(distance=0.5)

        self._setup_visual_objects_app2()

    def _setup_visual_objects_app1(self):
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

        # Rejilla de referencia
        grid = gl.GLGridItem()
        grid.setSize(x=2, y=2, z=2)
        grid.setSpacing(x=1, y=1, z=1)
        self.view.addItem(grid)

        # Ejes
        axes = gl.GLAxisItem()
        axes.setSize(4, 4, 4)
        self.view.addItem(axes)

        # vector de gravedad
        
        self.vector_grav = gl.GLLinePlotItem(pos=np.array([self.start_gravedad, self.end_gravedad]),
                                             color=(1, 1, 1, 1),
                                             width=4, mode='lines')
        #self.view.addItem(self.vector_grav)

        # vector de aceleración
        
        
        self.vector_acc = gl.GLLinePlotItem(
            pos=self.vector_acc,
            color=(1, 0.5, 0, 1),  # Color naranja
            width=4,             # Grosor de la línea
            mode='lines'
        )
        #self.view.addItem(vector_acc)

        # Tamaño de la caja
        box_size = (2, 0.05, 0.05)
        vertices  = create_box(size=box_size)
        # # Transformar al sistema NED (invertir el eje Z, intercambiar X e Y)
        self.vertices_ned = vertices.copy()
        self.vertices_ned[:, 2] = -vertices[:, 2]
        self.vertices_ned[:, 1] = -vertices[:, 1]

        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4]
        ])

        # Crear el objeto Mesh
        self.meshdata = gl.MeshData(vertexes=vertices, faces=faces)
        self.box = gl.GLMeshItem(meshdata=self.meshdata, color=(0, 1, 1, 0.5), smooth=False, drawEdges=True, edgeColor=(1, 0, 0, 1))
        self.view.addItem(self.box)

    def _setup_visual_objects_app2(self):
        axis_2 = gl.GLAxisItem()
        axis_2.setSize(10, 10, 10)
        self.window_2.addItem(axis_2)

        self.point = gl.GLScatterPlotItem()
        self.window_2.addItem(self.point)

    def parse_sensor_data(self, data_string, ref):
        clean_string = data_string.strip()
        data_split = clean_string.split(',')
        try:
            data = [float(value) for value in data_split]
            if len(data) == (9 * self.CANT_SAMPLES + 1):
                return data[ref:ref+3], data[ref+3:ref+6], data[ref+6:ref+9], data[ref+9]  # Acelerómetro, Giroscopio, Magnetómetro, Tiempo en ms
            else:
                return None
        except:
            return None

    def receive_data(self):
        data = b''
        while not data.endswith(b'\n'):
            packet = self.client_socket.recv(1)
            if not packet:
                return None
            data += packet
        return data.decode()

    def update(self):
        raw_data = self.receive_data()
        if not raw_data:
            return

        sensor_data = self.parse_sensor_data(raw_data, 0)
        if sensor_data is not None:
            self.acc, self.gyro, self.mag, milisegundos = sensor_data

            # # Aplicar rotaciones
            buff = self.acc[1]
            self.acc[1] = -self.acc[0]
            self.acc[0] = -buff
            self.acc = [x * 9.8 for x in self.acc]

            buff = self.gyro[1]
            self.gyro[1] = self.gyro[0]
            self.gyro[0] = buff
            self.gyro[2]= -self.gyro[2]
            self.gyro = np.radians(self.gyro)  # Convertir giroscopio a rad/s

            buff = self.mag[0]
            self.mag[0] = -self.mag[1]
            self.mag[1] = buff

            # Actualizar cuaternión con EKF
            self.Q = self.ekf.update(self.Q_buff, self.gyro, self.acc, self.mag, self.dt)
            self.Q_buff = self.Q

            # Actualizar visualización del vector de gravedad
            rotated_end_gravedad = self.rotate_vector_by_quaternion(self.end_gravedad, self.Q)
            rotated_vector_gravedad = np.array([self.start_gravedad, rotated_end_gravedad])
            self.vector_grav.setData(pos=rotated_vector_gravedad)

            # Rotar el vector de aceleracion
            new_vector_acc = np.array([self.start_gravedad, self.acc])
            self.vector_acc.setData(pos=new_vector_acc)

            # Rotar los vértices de la caja
            vertices_rotados = self.rotate_points(self.vertices_ned, self.Q)
            # Volver al sistema original (invertir el eje Z, intercambiar X e Y)
            vertices_rotados_buff = vertices_rotados.copy()
            vertices_rotados_buff[:, 2] = -vertices_rotados[:, 2]
            vertices_rotados_buff[:, 1] = -vertices_rotados[:, 1]
            # Actualizar la geometría de la caja en el gráfico
            self.meshdata.setVertexes(vertices_rotados_buff)
            self.box.meshDataChanged()

            #filtrado
            self.acc = self.filtro_pb.filter(self.acc)

            self.acc_global = self.remove_gravity(self.acc, self.Q)

            # 1. Predicción
            u = self.acc_global.reshape(3, 1)                # Entrada de control (aceleración medida)
            x_t = np.dot(self.F, self.x)                          # Predicción del estado
            P_t = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q_efk     # Predicción de la covarianza

            # 2. Actualización
            z = u - np.dot(self.H, x_t)

            # Ganancia de Kalman
            S = np.dot(np.dot(self.H, P_t), self.H.T) + self.R
            K = np.dot(np.dot(P_t, self.H.T), np.linalg.inv(S))

            # Corrección del estado
            self.x = x_t + np.dot(K, z)

            # Actualización de la covarianza
            self.P = P_t - np.dot(np.dot(K, self.H), P_t)

            # Simplificar el estado estimado
            self.x_estimado = self.x.flatten()

            # Convertir las estimaciones en arrays para graficar
            self.x_estimado = np.array(self.x_estimado)

            # Imprimir resultados
            self.update_point(self.x_estimado[0], self.x_estimado[1], self.x_estimado[2])

    def quaternion_conjugate(self,q):
        """
        Calcula el conjugado de un cuaternión.
        """
        w, x, y, z = q
        return [w, -x, -y, -z]

    def quaternion_multiply(self, q1, q2):
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

    def quaternion_inverse(self, q):
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

    def rotate_vector_by_quaternion(self, vector, quaternion):
        vector_quat = [0, 0, 0, 0]
        vector_quat[3] = vector[2]
        vector_quat[2] = vector[1]
        vector_quat[1] = vector[0]

        # Calcular el conjugado del cuaternión
        quat_conjugate = self.quaternion_conjugate(quaternion)

        # Rotar el vector: q * v * q^-1
        temp_result = self.quaternion_multiply(quaternion, vector_quat)
        rotated_quat = self.quaternion_multiply(temp_result, quat_conjugate)

        # Extraer la parte vectorial del cuaternión resultante
        rotated_vector = rotated_quat[1:]  # Ignorar la parte escalar
        return rotated_vector

    def rotate_points(self,points, quaternion):
        w, x, y, z = quaternion
        # Matriz de rotación derivada del cuaternión
        R = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])
        # Aplicar la rotación
        return np.dot(points, R.T)

    def remove_gravity(self, acc, Q):
        if (np.linalg.norm(acc) < 9.9) and (np.linalg.norm(acc) > 9.7):
            return np.array([0.0, 0.0, 0.0])
        Q = self.quaternion_inverse(Q)
        gravity_corr = np.array(self.rotate_vector_by_quaternion(self.gravity_global, Q))
        acc_corr = acc - gravity_corr
        return acc_corr

    def update_point(self, x, y, z):
        data = np.array([[x, y, z]])
        self.point.setData(pos=data, size=10, color=(1, 0, 0, 1))

    def run(self):
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(self.dt * 1000))

        self.view.show()
        sys.exit(self.app.exec_())

# Uso como módulo principal
if __name__ == "__main__":
    esp32_ip = '192.168.1.71'  # Dirección IP del ESP32
    esp32_port = 80             # Puerto del ESP32

    visualizer = IMUVisualizer(esp32_ip, esp32_port)
    visualizer.run()
