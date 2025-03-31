import numpy as np
from ahrs.filters import EKF
import csv
import time

class IMUVisualizer:

    def __init__(self, u_ia_ori = [0,0,0,0], u_ia_pos = [0,0,0,0], dt=0.005):
        self.dt = dt
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
        self.P_pos = np.eye(6) * 5  # Incertidumbre inicial en posición y velocidad
        self.P_fus_ori = np.eye(4) * 1 # Incertidumbre inicial de la posición

        # Matriz de covarianza del proceso (Q): incertidumbre del modelo
        self.Q_pos = np.eye(6) * 10
        self.Q_fus_ori = np.eye(4) * 0.5

        # Matriz de covarianza de las mediciones (R): incertidumbre del sensor
        self.R_pos = np.eye(3) *1
        self.R_fus_ori_ia = np.eye(4) * 0.1
        self.R_fus_ori_sensor = np.eye(4) * 0.8

        # Matriz de medición (H):
        self.H_pos = np.array([ [1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0]])
        self.H_fus_ori = np.array([ [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        # Inicialización de sensores
        self.acc = np.zeros(3)
        self.acc_prev = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)
        self.Q = np.zeros((1, 4))  # Allocate array for quaternions
        self.Q_buff = np.array([0.7071, 0, 0.7071, 0])

        # Configuración del filtro EKF
        self.ekf = EKF(frequency=(1/self.dt), magnetic_ref=-40.0, noises=[0.4**2, 0.05**2, 0.2**2], P=(np.identity(4)*100), q0=self.Q_buff)


    def parse_sensor_data(self, data_string, ref):
        clean_string = data_string.strip()
        data_split = clean_string.split(',')
        try:
            data = [float(value) for value in data_split]
            if len(data) == (22):
                return {'acc_1'     :   data[0:3], 
                        'gyro_1'    :   data[3:6], 
                        'mag_1'     :   data[6:9], 
                        'acc_2'     :   data[9:12], 
                        'gyro_2'    :   data[12:15], 
                        'mag_2'     :   data[15:18], 
                        'boton_2'   :   data[18],
                        'pal_indic' :   data[19],
                        'timestamp' :   data[20], 
                        'boton_1'   :   data[21]}
            elif len(data) == (12):
                return {'acc_1'     :   data[0:3], 
                        'gyro_1'    :   data[3:6], 
                        'mag_1'     :   data[6:9], 
                        'acc_2'     :   None, 
                        'gyro_2'    :   None, 
                        'mag_2'     :   None, 
                        'boton_2'   :   1,
                        'pal_indic' :   data[9],
                        'timestamp' :   data[10], 
                        'boton_1'   :   data[11]}
            else:
                return None
        except:
            return None

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
        self.Q_buff = self.Q_buff[:,0]
        self.Q_buff = self.Q_buff / np.linalg.norm(self.Q_buff)

    def posicion_fus_kf(self):
        # 1. Predicción
        acc = np.array(self.acc)
        u = self.remove_gravity(acc, self.Q)
        if u.size == 9:
            u =  u[:,0]
        #print("u: ", u)
        u = u.reshape(3, 1)

        # 1. Predicción
        x_t = np.dot(self.F_pos, self.x_pos) + np.dot(self.B_pos, u)                 # Predicción del estado
        P_t = np.dot(np.dot(self.F_pos, self.P_pos), self.F_pos.T) + self.Q_pos      # Predicción de la covarianza

        if not np.all(self.u_ia_pos == 0):
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
        else:
            # Corrección del estado
            self.x_pos = x_t
            # Actualización de la covarianza
            self.P_pos = P_t
    
        self.x_pos =  self.x_pos[:,0]
        self.x_pos = self.x_pos.reshape(6,1)

        return self.x_pos

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

    def guardar_cuaternion_csv(self, nombre_archivo, datos_cuaternion):
        with open(nombre_archivo, mode='w', newline='') as archivo:
            escritor = csv.writer(archivo, delimiter=';')
            escritor.writerow(["timestamp", "X", "Y", "Z"])
            escritor.writerows(datos_cuaternion)

    def update_kf(self, u_ia_ori = [0,0,0,0], u_ia_pos = [0,0,0], gyro = [0,0,0], mag = [0,0,0], acc = [0,0,0]):
        self.u_ia_ori = u_ia_ori
        self.u_ia_pos = u_ia_pos

        self.acond_info_imu(gyro, mag, acc)
    ################################################################## Kalman de Orientación
        self.Q = self.ekf.update(self.Q_buff, self.gyro, self.acc, self.mag, self.dt)
        self.Q_buff = self.Q
    ################################################################## Kalman de Orientación Fusión
        if not (np.all(self.u_ia_ori == 0)):
            self.ori_fus_kf()
    ################################################################## Kalman de Posición Sensor y Fusión
        self.x_estimado = self.posicion_fus_kf()                                          