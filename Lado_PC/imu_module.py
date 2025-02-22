import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy.spatial.transform import Rotation as R
import socket
import sys
from PyQt5 import QtWidgets
from collections import deque
from ahrs.filters import EKF
import asyncio


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

    def __init__(self, esp32_ip, esp32_port, u_ia_ori = [0,0,0,0], u_ia_pos = [0,0,0,0], dt=0.005):
        self.CANT_SAMPLES = 1
        self.dt = dt
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.u_ia_ori = u_ia_ori
        self.u_ia_pos = u_ia_pos

        # Referencia de acelerómetro
        self.acc_ref = np.array([0, 0, 1])
        # Gravedad en el sistema global
        self.gravity_global = np.array([0.0, 0.0, 9.8])

        # Matrices del sistema
        # Estado inicial: posición y velocidad [x, y, z, vx, vy, vz]
        self.x_pos = np.array([[0], [0], [0], [0], [0], [0]])  # Estado inicial: posición y velocidad en 3D
        self.x_acc = np.array([[0], [0], [1]])
        self.x_fus_ori = np.array([[0], [0], [0], [0]])
        self.x_estimado = np.zeros(6)

        # Matriz de transición de estado (F)
        self.F_pos = np.array([ [1, 0, 0, self.dt,  0,          0       ],
                                [0, 1, 0, 0,        self.dt,    0       ],
                                [0, 0, 1, 0,        0,          self.dt ],
                                [0, 0, 0, 1,        0,          0       ],
                                [0, 0, 0, 0,        1,          0       ],
                                [0, 0, 0, 0,        0,          1       ]])
        self.F_fus_ori = np.array([ [1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,0],
                                    [0,0,0,1]])

        # Matriz de control (B) para incluir aceleración medida
        self.B_pos = np.array([ [0.5 * self.dt**2,  0,                  0               ],
                                [0,                 0.5 * self.dt**2,   0               ],
                                [0,                 0,                  0.5 * self.dt**2],
                                [self.dt,           0,                  0               ],
                                [0,                 self.dt,            0               ],
                                [0,                 0,                  self.dt         ]])

        # Matriz de covarianza inicial (P)
        self.P_pos = np.eye(6) * 1  # Incertidumbre inicial en posición y velocidad
        self.P_fus_ori = np.eye(4) * 5 # Incertidumbre inicial de la posición

        # Matriz de covarianza del proceso (Q): incertidumbre del modelo
        self.Q_pos = np.eye(6) * 0.8
        self.Q_fus_ori = np.eye(4) * 0.5

        # Matriz de covarianza de las mediciones (R): incertidumbre del sensor
        self.R_pos = np.eye(3) * 0.1
        self.R_fus_ori_ia = np.eye(4) * 0.01
        self.R_fus_ori_sensor = np.eye(4) * 0.1

        # Matriz de medición (H):
        self.H_pos = np.array([ [1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0]])
        self.H_fus_ori = np.array([ [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

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
        self.ekf = EKF(frequency=(1/self.dt), magnetic_ref=-40.0, noises=[0.4**2, 0.05**2, 0.2**2], P=(np.identity(4)*100), q0=self.Q_buff)

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
        self.view.addItem(self.vector_grav)

        # vector de aceleración
        self.vector_acc = gl.GLLinePlotItem(
            pos=self.vector_acc,
            color=(1, 0.5, 0, 1),  # Color naranja
            width=4,             # Grosor de la línea
            mode='lines'
        )
        self.view.addItem(self.vector_acc)

        # Tamaño de la caja
        box_size = (2, 0.05, 0.05)
        vertices  = create_box(size=box_size)
        # Transformar al sistema NED (invertir el eje Z, intercambiar X e Y)
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
            if len(data) == (9 * self.CANT_SAMPLES + 2):
                return data[ref:ref+3], data[ref+3:ref+6], data[ref+6:ref+9], data[ref+9], data[ref+10]  # Acelerómetro, Giroscopio, Magnetómetro, Tiempo en ms
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

    def acond_info_imu(self, gyro = [0,0,0], mag = [0,0,0], acc = [0,0,0]):
        self.acc = acc
        self.gyro = gyro
        self.mag = mag
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

    def acond_info_graf(self):
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

    def ori_fus_kf(self):
        u_sensor_ori = self.Q
        # 1. Predicción
        x_t = np.dot(self.F_fus_ori, self.x_fus_ori)                                                # Predicción del estado
        P_t = np.dot(np.dot(self.F_fus_ori, self.P_fus_ori), self.F_fus_ori.T) + self.Q_fus_ori     # Predicción de la covarianza

        # 2. Actualización
        # Innovación (residuo): diferencia entre medición y predicción
        z = self.u_ia_ori - np.dot(self.H_fus_ori, x_t)

        # Ganancia de Kalman
        S = np.dot(np.dot(self.H_fus_ori, P_t), self.H_fus_ori.T) + self.R_fus_ori_ia
        K = np.dot(np.dot(P_t, self.H_fus_ori.T), np.linalg.inv(S))

        # Corrección del estado
        self.x_fus_ori = x_t + np.dot(K, z)

        # Actualización de la covarianza
        self.P_fus_ori = P_t - np.dot(np.dot(K, self.H_fus_ori), P_t)

        # 2. Actualización
        # Innovación (residuo): diferencia entre medición y predicción
        z = u_sensor_ori - np.dot(self.H_fus_ori, self.x_fus_ori)

        # Ganancia de Kalman
        S = np.dot(np.dot(self.H_fus_ori, self.P_fus_ori), self.H_fus_ori.T) + self.R_fus_ori_sensor
        K = np.dot(np.dot(self.P_fus_ori, self.H_fus_ori.T), np.linalg.inv(S))

        # Corrección del estado
        self.x_fus_ori = self.x_fus_ori + np.dot(K, z)

        # Actualización de la covarianza
        self.P_fus_ori = self.P_fus_ori - np.dot(np.dot(K, self.H_fus_ori), self.P_fus_ori)

        self.Q_buff = self.x_fus_ori

    def posicion_fus_kf(self):
        #filtrado
        # acc = self.filtro_pb.filter(acc)
        # acc = self.filtro_mm.filter(acc)

        # 1. Predicción
        acc = np.array(self.acc)
        u = self.remove_gravity(acc, self.Q)
        u = u.reshape(3, 1)

        # 1. Predicción
        x_t = np.dot(self.F_pos, self.x_pos) + np.dot(self.B_pos, u)                 # Predicción del estado
        P_t = np.dot(np.dot(self.F_pos, self.P_pos), self.F_pos.T) + self.Q_pos      # Predicción de la covarianza

        # 2. Actualización
        # Innovación (residuo): diferencia entre medición y predicción
        z = self.u_ia_pos - np.dot(self.H_pos, x_t)

        # Ganancia de Kalman
        S = np.dot(np.dot(self.H_pos, P_t), self.H_pos.T) + self.R_pos
        K = np.dot(np.dot(P_t, self.H_pos.T), np.linalg.inv(S))

        # Corrección del estado
        self.x_pos = x_t + np.dot(K, z)

        # Actualización de la covarianza
        self.P_pos = P_t - np.dot(np.dot(K, self.H_pos), P_t)

        # # Simplificar el estado estimado
        # x_estimado = self.x_pos.flatten()
        # # Convertir las estimaciones en arrays para graficar
        # x_estimado = np.array(x_estimado)

        return self.x_pos

    def rotate_vector_by_quaternion(self, vector, q):

        R_matrix = np.array([
            [q[0]**2+q[1]**2-q[2]**2-q[3]**2,   2.0*(q[1]*q[2]-q[0]*q[3]),          2.0*(q[1]*q[3]+q[0]*q[2])],
            [2.0*(q[1]*q[2]+q[0]*q[3]),         q[0]**2-q[1]**2+q[2]**2-q[3]**2,    2.0*(q[2]*q[3]-q[0]*q[1])],
            [2.0*(q[1]*q[3]-q[0]*q[2]),         2.0*(q[0]*q[1]+q[2]*q[3]),          q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
        
        buff = R_matrix @ vector
        rotated_vector = buff
        #rotated_vector = buff[1:]  # Ignorar la parte escalar
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

    # Función para corregir el efecto de la gravedad
    def remove_gravity(self, acc, q):
        if (np.linalg.norm(acc) < 9.9) and (np.linalg.norm(acc) > 9.7):
            return np.array([0.0, 0.0, 0.0])

        R_matrix = np.array([
            [q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
            [2.0*(q[1]*q[2]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2.0*(q[2]*q[3]-q[0]*q[1])],
            [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
        
        gravity_corr = R_matrix @ self.gravity_global
        acc_corr = acc - gravity_corr
        return acc_corr

    def H_matrix(self, q, x):
        R_matrix = np.array([
            [q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
            [2.0*(q[1]*q[2]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2.0*(q[2]*q[3]-q[0]*q[1])],
            [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
        
        acc_rot = R_matrix @ x
        acc_rot = np.array(acc_rot)
        return acc_rot

    def update_point(self, x, y, z):
        data = np.array([[x, y, z]])
        self.point.setData(pos=data, size=10, color=(1, 0, 0, 1))


    def update_kf(self, u_ia_ori = [0,0,0,0], u_ia_pos = [0,0,0], gyro = [0,0,0], mag = [0,0,0], acc = [0,0,0]):
        self.u_ia_ori = u_ia_ori
        self.u_ia_pos = u_ia_pos

        if __name__ == "__main__":
            raw_data = self.receive_data()
            if not raw_data:
                return
            sensor_data = self.parse_sensor_data(raw_data, 0)
            self.acc, self.gyro, self.mag, _, _ = sensor_data
            
        self.acond_info_imu(gyro, mag, acc)
################################################################## Kalman de Orientación
        self.Q = self.ekf.update(self.Q_buff, self.gyro, self.acc, self.mag, self.dt)
        self.Q_buff = self.Q
################################################################## Kalman de Orientación Fusión
        # self.ori_fus_kf()
################################################################## Kalman de Posición Sensor y Fusión
        self.x_estimado = self.posicion_fus_kf()                                            

    def run(self):
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(self.dt * 500))

        self.view.show()
        sys.exit(self.app.exec_())

# Uso como módulo principal
if __name__ == "__main__":
    esp32_ip = '192.168.1.71'  # Dirección IP del ESP32
    esp32_port = 80             # Puerto del ESP32
    dt = 0.02
    u_ia_ori = np.array([0, 0, 0, 0])
    u_ia_pos = np.array([0, 0, 0])

    visualizer = IMUVisualizer(esp32_ip, esp32_port, u_ia_ori, u_ia_pos, dt=dt)
    visualizer.run()