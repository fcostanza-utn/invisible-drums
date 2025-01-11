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
    def __init__(self, esp32_ip, esp32_port, dt=0.005):
        self.dt = dt
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port

        # Conexión con ESP32
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.esp32_ip, self.esp32_port))

        # Inicialización de sensores
        self.acc = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)
        self.q = np.array([1, 0, 0, 0])  # Cuaternión inicial

        # Configuración del filtro EKF
        self.ekf = EKF(frequency=1 / self.dt, magnetic_ref=-40.0, 
                       noises=[0.4**2, 0.05**2, 0.2**2],
                       P=np.identity(4) * 100, q0=self.q)

        # Crear ventana 3D
        self.app = QtWidgets.QApplication([])
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('3D Visualization')
        self.view.setCameraPosition(distance=6, azimuth=90)

        # Configurar objetos visuales
        self._setup_visual_objects()

    def _setup_visual_objects(self):
        # Rejilla de referencia
        grid = gl.GLGridItem()
        grid.setSize(x=2, y=2, z=2)
        grid.setSpacing(x=1, y=1, z=1)
        self.view.addItem(grid)

        # Ejes
        axes = gl.GLAxisItem()
        axes.setSize(4, 4, 4)
        self.view.addItem(axes)

        # Crear vector de gravedad
        start_gravedad = np.array([0, 0, 0])
        end_gravedad = np.array([0, 0, 1])
        self.vector_grav = gl.GLLinePlotItem(pos=np.array([start_gravedad, end_gravedad]),
                                             color=(1, 1, 1, 1),
                                             width=4, mode='lines')
        self.view.addItem(self.vector_grav)

    def parse_sensor_data(self, data_string):
        try:
            data = [float(value) for value in data_string.split(',')]
            if len(data) >= 9:
                return data[:3], data[3:6], data[6:9]
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

    def update_visualization(self):
        raw_data = self.receive_data()
        if not raw_data:
            return

        sensor_data = self.parse_sensor_data(raw_data)
        if sensor_data is not None:
            self.acc, self.gyro, self.mag = sensor_data

            # Actualizar cuaternión con EKF
            self.q = self.ekf.update(self.q, np.radians(self.gyro), self.acc, self.mag, self.dt)

            # Actualizar visualización del vector de gravedad
            rotated_gravity = self.rotate_vector([0, 0, 1], self.q)
            self.vector_grav.setData(pos=np.array([[0, 0, 0], rotated_gravity]))

    def rotate_vector(self, vector, quaternion):
        rotation = R.from_quat(quaternion)
        return rotation.apply(vector)

    def run(self):
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(int(self.dt * 1000))

        self.view.show()
        sys.exit(self.app.exec_())

# Uso como módulo principal
if __name__ == "__main__":
    esp32_ip = '192.168.0.186'  # Dirección IP del ESP32
    esp32_port = 80             # Puerto del ESP32

    visualizer = IMUVisualizer(esp32_ip, esp32_port)
    visualizer.run()
