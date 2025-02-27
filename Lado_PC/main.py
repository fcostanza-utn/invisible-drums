#librerías
from ultralytics import YOLO
import logging
import cv2
from openni import openni2
import math
from collections import deque
import mido
from imu_module import IMUVisualizer
from PyQt5 import QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys
import time
import torch
import numpy as np
from data_sync import DataSynchronizer
from queue import Queue, Empty
import threading

"""
Variables de datos y control
"""
init_meseaurement = False
processing_flag = True

data_sync = DataSynchronizer()      # Instancia compartida para sincronizar datos entre hilos

stop_event = threading.Event()      # Creamos un evento que actuará como bandera para detener el hilo

tip_points = []
mid_points = []

blue_tip_detection = None
red_tape_detection = None

mtx_rgb = np.array([[549.512901,    0.,             303.203858],
                    [0.,            550.614039,     232.135452],
                    [0.,            0.,             1.]])

mtx_ir = np.array([ [622.443923,    0.,             301.653252],
                    [0.,            623.759527,     232.812601],
                    [0.,            0.,             1.]])

mtx_prof = np.array([   [0,    0,     0],
                        [0,    0,     0],
                        [0,    0,     0]])

tras_vector_prof = np.array([[0.05950176],
                             [0.02782342],
                             [0.89904015]])

Rot_matrix_prof = np.array([[0.46891086,     -0.88155789,    -0.05457372],
                            [0.87468821,     0.45490053,     0.16729026],
                            [-0.12265043,    -0.12617921,    0.9843961]])

T_prof = np.block([ [Rot_matrix_prof,   tras_vector_prof],
                    [np.zeros((1, 3)),  np.ones((1, 1)) ]])

tras_vector_stereo = np.array([[2.368586e-02],
                               [2.909582e-04],
                               [4.632117e-04]])

Rot_matrix_stereo = np.array([[9.9997269257755e-01,     6.7448757651869e-03,    -3.0200579643581e-03],
                              [-6.7584186667168e-03,    9.9996705075785e-01,    -4.4967961679709e-03],
                              [2.9896281242425e-03,     4.5170841881792e-03,    9.9998532892944e-01]])

contador_grafico = 0

# Configuración graficos
num_muestras = 100  # Número de muestras visibles en el eje X
x_data = deque(maxlen=num_muestras)
y_data = deque(maxlen=num_muestras)
z_data = deque(maxlen=num_muestras)
t_data = deque(maxlen=num_muestras)  # Contador de muestras (eje X)
t_data = deque(maxlen=num_muestras)  # Contador de muestras (eje X)

"""
Colas de Datos
"""
sensor_queue = Queue(maxsize=100)   # Para datos crudos de la IMU
camara_queue = Queue(maxsize=100)   # Para datos crudos de la cámara
graf_queue = Queue(maxsize=100)   # Para datos crudos para graficar

"""
INICIALIZACION DE YOLO y Midas
"""
# modelo YOLO
# modelo YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO("yolo-Weights/best_yolo11m_vKinect.pt").to(device)
classNames = ["drumsticks_mid", "drumsticks_tip"] # Definir las clases de objetos para la detección

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Iniciar la cámara
openni2.initialize("C:/Program Files/OpenNI2/Redist")
modeRGB = openni2.VideoMode(
        pixelFormat=openni2.PIXEL_FORMAT_RGB888,  #formato de 24 bits
        resolutionX=640,
        resolutionY=480,
        fps=60
    )
modeIR = openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_GRAY16,  #formato de 16 bits
            resolutionX=640,
            resolutionY=480,
            fps=60
        )
device_ni = openni2.Device.open_any()
depth_stream = device_ni.create_depth_stream()
color_stream = device_ni.create_color_stream()
color_stream.set_video_mode(modeRGB)
depth_stream.set_video_mode(modeIR)
depth_stream.start()
color_stream.start()

"""
INICIALIZACION DE IMU
"""
visualizer = IMUVisualizer('192.168.1.71', 80, 0.02)
"""
INICIALIZACION DE MIDI
"""
print("Available MIDI output ports:")
print(mido.get_output_names())
portmidi = mido.Backend('mido.backends.rtmidi')
midi_out = portmidi.open_output('MIDI1 1')

'''
Graficos QT
'''
# # Crear una aplicación de PyQt
# app = QtWidgets.QApplication(sys.argv)
# window = gl.GLViewWidget()
# window.show()
# window.setWindowTitle('Gráfico 3D en Tiempo Real')
# window.setCameraPosition(distance=0.5)
# # Crear los ejes
# axis = gl.GLAxisItem()
# axis.setSize(10, 10, 10)
# window.addItem(axis)
# # Crear el punto en 3D
# point = gl.GLScatterPlotItem()
# window.addItem(point)

# # Configuración inicial del punto
# def update_point(x, y, z):
#     data = np.array([[x, y, z]])
#     data = np.squeeze(data)
#     point.setData(pos=data, size=10, color=(1, 0, 0, 1))

"""
FUNCIONES
"""
# Función para enviar una nota MIDI
def send_midi_note(note, acc):
    if note:
        # if 0 < (np.linalg.norm(acc) - 9.8) < 10:
        #     midi_out.send(mido.Message('note_on', note=note, velocity=15))
        #     midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve
        # elif 10 < (np.linalg.norm(acc) - 9.8) < 20:
        #     midi_out.send(mido.Message('note_on', note=note, velocity=30))
        #     midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve
        # elif 20 < (np.linalg.norm(acc) - 9.8) < 30:
        #     midi_out.send(mido.Message('note_on', note=note, velocity=45))
        #     midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve
        # elif 30 < (np.linalg.norm(acc) - 9.8) < 40:
        #     midi_out.send(mido.Message('note_on', note=note, velocity=60))
        #     midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve
        # elif 40 < (np.linalg.norm(acc) - 9.8) < 50:
        #     midi_out.send(mido.Message('note_on', note=note, velocity=75))
        #     midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve
        # elif 50 < (np.linalg.norm(acc) - 9.8) < 60:
        #     midi_out.send(mido.Message('note_on', note=note, velocity=90))
        #     midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve
        # elif 60 < (np.linalg.norm(acc) - 9.8) < 70:
        midi_out.send(mido.Message('note_on', note=note, velocity=100))
        midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve

def map_position_to_midi(x, y, z):
    x = x * 100
    y = y * 100
    z = z * 100
    # print(f"Posicion midi: {float(x):.1f} {float(y):.1f} {float(z):.1f}")
    if (-25 < x < 5) and (45 < y < 50) and (-30 < z < 0):
        return 38  # Nota MIDI para un snare drum
    elif (-50 < x < -20) and (15 < y < 20) and (-25 < z < 5):
        return 42  # Nota MIDI para un hihat drum
    elif (-50 < x < -20) and (0 < y < 5) and (-60 < z < -30):
        return 49  # Nota MIDI para un crash drum
    elif (20 < x < 50) and (15 < y < 20) and (-55 < z < -15):
        return 51  # Nota MIDI para un ride drum
    elif (-30 < x < 0) and (20 < y < 25) and (-65 < z < -35):
        return 50  # Nota MIDI para un hightom drum
    elif (15 < x < 45) and (50 < y < 55) and (-35 < z < -5):
        return 45  # Nota MIDI para un lowtom drum
    return None

def corregir_mapa_profundidad(depth_frame, T, fx, fy, cx, cy):
    """
    Corrige el mapa de profundidad aplicando la transformación T de forma vectorizada.
    
    Parámetros:
      depth_frame: arreglo 2D de profundidad (por ejemplo, 480x640) en unidades (milímetros)
      T: matriz de transformación 4x4 (numpy.array) que compensa el offset entre proyector y cámara IR.
      fx, fy: focales en píxeles
      cx, cy: coordenadas del centro óptico
      
    Retorna:
      nuevo_depth: mapa de profundidad corregido (2D, mismo tamaño que depth_frame)
    """
    H, W = depth_frame.shape
    # Crear rejilla de coordenadas para cada píxel
    grid_y, grid_x = np.indices((H, W))  # grid_y y grid_x tienen forma (H, W)
    
    # Aplanar los arreglos para operar vectorizadamente
    grid_x = grid_x.flatten()  # forma (N,)
    grid_y = grid_y.flatten()
    depth = depth_frame.flatten()  # forma (N,)
    
    # Filtrar píxeles con profundidad válida (no cero)
    valid = depth > 0
    grid_x = grid_x[valid]
    grid_y = grid_y[valid]
    depth = depth[valid]
    
    # Convertir coordenadas de imagen (projective) a coordenadas reales 3D
    X = (grid_x - cx) * depth / fx
    Y = (grid_y - cy) * depth / fy
    Z = depth
    
    # Formar las coordenadas homogéneas (N, 4)
    points = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)
    
    # Aplicar la transformación T a todos los puntos a la vez
    points_corr = (T @ points.T).T  # forma (N, 4)
    
    # Reproyección a coordenadas de imagen: x' = (X'/Z') * fx + cx, y' = (Y'/Z') * fy + cy
    X_corr = points_corr[:, 0]
    Y_corr = points_corr[:, 1]
    Z_corr = points_corr[:, 2]
    
    # Evitar división por cero (Z_corr no debe ser 0 en condiciones normales)
    new_x = np.round((X_corr / Z_corr) * fx + cx).astype(int)
    new_y = np.round((Y_corr / Z_corr) * fy + cy).astype(int)
    
    # Filtrar los puntos que caen dentro de los límites de la imagen
    valid_idx = (new_x >= 0) & (new_x < W) & (new_y >= 0) & (new_y < H)
    new_x = new_x[valid_idx]
    new_y = new_y[valid_idx]
    Z_corr = Z_corr[valid_idx]
    
    # Crear un mapa de profundidad nuevo, inicializado con infinito para la operación de mínimo
    new_depth_flat = np.full(H * W, np.inf, dtype=np.float32)
    # Calcular índices planos de los píxeles reproyectados
    flat_idx = new_y * W + new_x
    
    # Utilizar scatter-min: para cada índice, asignar el menor valor de Z_corr
    np.minimum.at(new_depth_flat, flat_idx, Z_corr)
    
    # Reemplazar los valores que quedaron en inf (sin asignación) por 0
    new_depth_flat[new_depth_flat == np.inf] = 0.0
    # Remodelar el arreglo a la forma original de la imagen
    nuevo_depth = new_depth_flat.reshape(H, W)
    
    return nuevo_depth

def aprox_depth_disp(depth_frame, Xp, Yp):
    disparidad = 50
    Xp_min = max(Xp - disparidad, 0)
    Xp_max = min(Xp + disparidad, depth_frame.shape[1])  # Asegurar que no exceda las columnas
    # Extraer valores dentro de los límites
    z_value = depth_frame[Yp, Xp_min:Xp_max]
    z_value = z_value[z_value != 0]

    if z_value.size > 0:
        # Obtener el valor mínimo y convertirlo a entero
        z_value_min = int(np.min(z_value))
        return z_value_min
    else:
        disparidad = 100
        Xp_min = max(Xp - disparidad, 0)
        Xp_max = min(Xp + disparidad, depth_frame.shape[1])  # Asegurar que no exceda las columnas
        # Extraer valores dentro de los límites
        z_value = depth_frame[Yp, Xp_min:Xp_max]
        z_value = z_value[z_value != 0]
        if z_value.size > 0:
            # Obtener el valor mínimo y convertirlo a entero
            z_value_min = int(np.min(z_value))
            return z_value_min
        else:
            disparidad = 200
            Xp_min = max(Xp - disparidad, 0)
            Xp_max = min(Xp + disparidad, depth_frame.shape[1])  # Asegurar que no exceda las columnas
            # Extraer valores dentro de los límites
            z_value = depth_frame[Yp, Xp_min:Xp_max]
            z_value = z_value[z_value != 0]
            if z_value.size > 0:
                # Obtener el valor mínimo y convertirlo a entero
                z_value_min = int(np.min(z_value))
                return z_value_min
    return 0

##########################
# 1. Hilo de captura (sensor)
##########################
def sensor_capture_thread():
    while not stop_event.is_set():
        start_time = time.time()
        imu_data = visualizer.receive_data()  # Leer IMU
        if not imu_data:
            time.sleep(0.005)  # 5ms entre muestras
        else: 
            if data_sync.get_state()['button']:
                _, _, _, ref_time, ref_but = visualizer.parse_sensor_data(imu_data, 0)
                data_sync.set_button(ref_but)
                data_sync.update_imu_time(ref_time)
            if not data_sync.get_state()['button']:
                if data_sync.get_state()['offset_time_camera'] == 0:
                    data_sync.update_camera_time(time.time())
                acc, gyro, mag, ref_time, ref_but = visualizer.parse_sensor_data(imu_data, 0)
                if ref_but == 0:
                    data_sync.set_button_repeat(True)
                    data_sync.update_camera_time(time.time())
                    data_sync.update_imu_time(ref_time)
                    ref_but = 1
                sensor_queue.put((acc, gyro, mag, ref_time))
                end_time = time.time()
                # print(f"Tiempo de procesamiento captura: {end_time - start_time:.5f} segundos")
            time.sleep(0.015)  # 10ms entre muestras

    
##########################
# 2. Hilo de procesamiento de imagen
##########################
def image_processing_thread():
    while not stop_event.is_set():
        start_time = time.time()

        color_frame = color_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        color_data = np.array(color_frame.get_buffer_as_triplet()).reshape((color_frame.height, color_frame.width, 3))    
        color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
        depth_corregido = np.array(depth_frame.get_buffer_as_uint16()).reshape((depth_frame.height, depth_frame.width))

        if color_data is not None:
            elapsed_time = time.time()
            results = model_yolo.predict(color_data, conf=0.4, stream=True)
            points = {}

            for r in results:
                # depth_corregido = corregir_mapa_profundidad(depth_data, T_prof, mtx_prof[0,0], mtx_prof[1,1], mtx_prof[0,2], mtx_prof[1,2])
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    Xp, Yp = (x1 + x2) // 2, (y1 + y2) // 2
                    # z_value = int(depth_corregido[Yp, Xp])
                    z_value = aprox_depth_disp(depth_corregido, Xp, Yp)

                    # XY_stereo = np.dot(Rot_matrix_stereo, np.array([[Xp], [Yp], [z_value]])) + tras_vector_stereo
                    # if int(XY_stereo[1,0]) > 480:
                    #     XY_stereo[1,0] = 480
                    # if int(XY_stereo[0,0]) > 640:
                    #     XY_stereo[0,0] = 640
                    # # z_value = int(depth_corregido[int(XY_stereo[1,0]), int(XY_stereo[0,0])])
                    # z_value = aprox_depth_disp(depth_corregido, XY_stereo[0,0], XY_stereo[1,0])

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    cv2.rectangle(color_data, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(color_data, f"{class_name} {confidence} Coord:({Xp},{Yp},{z_value})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    points[class_name] = (Xp, Yp, z_value)

            # Dibujar línea entre drumsticks_mid y drumsticks_tip
            if "drumsticks_mid" in points and "drumsticks_tip" in points:
                cv2.line(color_data, points["drumsticks_mid"][:2], points["drumsticks_tip"][:2], (0, 255, 0), 2)
                (X_blue, Y_blue, Z_blue) = points["drumsticks_tip"]
                (X_red, Y_red, Z_red) = points["drumsticks_mid"]
                if data_sync.get_state()['button']:
                    data_sync.set_offsets(X_blue, Y_blue, Z_blue)
                if data_sync.get_state()['button_repeat']:
                    data_sync.set_offsets(X_blue, Y_blue, Z_blue)
                    data_sync.set_button_repeat(False)  
                if not data_sync.get_state()['button']:
                    camara_queue.put((X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, elapsed_time))

            cv2.imshow("Cam", color_data)
            cv2.imshow("Depth", depth_corregido)

        end_time = time.time()
        # print(f"Tiempo de procesamiento camara: {end_time - start_time:.3f} segundos")
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()  # Señalizamos al hilo 1 que debe detenerse
            break
        time.sleep(0.005)  # 5ms entre muestras
    
    openni2.unload()
    cv2.destroyAllWindows()

##########################
# 3. Hilo de actualización del Kalman
##########################
def kalman_update_thread():
    flag_imu_empty = False
    flag_cam_empty = False
    camera_time = 0
    imu_time = 0
    u_ia_pos = np.zeros((3, 1))
    u_ia_ori = np.zeros((4, 1))
    acc = np.zeros((3, 1))
    note = None

    while not stop_event.is_set():
        start_time = time.time()
        try:
            acc, gyro, mag, milisegundos = sensor_queue.get(block=False)
            flag_imu_empty = False
        except Empty:
            flag_imu_empty = True
            # print("La cola sensor_queue está vacía.")
        try:
            X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, elapsed_time = camara_queue.get(block=False)
            flag_cam_empty = False
        except Empty:
            flag_cam_empty = True
            # print("La cola camara_queue está vacía.")

    ################################################# CALCULO DE ORIENTACIÓN MEDIANTE CÁMARA
        if not flag_cam_empty:
            vector_ori = np.array([X_blue - X_red, Y_blue - Y_red, Z_red - Z_blue])
            vector_ori = vector_ori/np.linalg.norm(vector_ori)

            u = np.array([0,0,1 ])
            dot = np.dot(u, vector_ori)

            # Caso especial: u y vector_ori son opuestos (rotación de 180°)
            if dot < -0.999999:
                # Elegimos un eje ortogonal arbitrario
                orth = np.cross(np.array([0, 0, 1]), u)
                if np.linalg.norm(orth) < 1e-6:
                    orth = np.cross(np.array([0, 1, 0]), u)
                orth = orth / np.linalg.norm(orth)
                qw = 0
                qx = orth[0]
                qy = orth[1]
                qz = orth[2]
            else: 
                axis = np.cross(u, vector_ori)
                axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else np.array([0, 0, 1])
                # Calcular el ángulo
                theta = math.acos(np.clip(dot, -1.0, 1.0))
                # Calcular el cuaternión
                s = math.sin(theta / 2)
                qw = math.cos(theta / 2)
                qx = axis[0] * s
                qy = axis[1] * s
                qz = axis[2] * s

    ################################################# ACTUALIZAR FILTROS DE KALMAN
        if not flag_imu_empty:
            imu_time = milisegundos - data_sync.get_state()['offset_time_imu']
            # print("imu_time: ", imu_time)
        
        if not flag_cam_empty:   
            camera_time = (elapsed_time - data_sync.get_state()['offset_time_camera'])*1000
            # print("camera_time: ", camera_time)

        if (camera_time > imu_time - 50) and (camera_time < imu_time + 50) and not flag_imu_empty:
            state = data_sync.get_state()
            if not flag_cam_empty:
                if X_blue and Y_blue and not Z_blue:
                    Z_blue = u_ia_pos[2] * 1000 + state['z_offset']
                else:
                    u_ia_pos[2] = (Z_blue - state['z_offset']) / 1000
                u_ia_pos[0] = (((X_blue - mtx_rgb[0,2]) * Z_blue - (state['x_offset'] - mtx_rgb[0,2]) * state['z_offset']) / mtx_rgb[0,0]) / 1000
                u_ia_pos[1] = (((Y_blue - mtx_rgb[1,2]) * Z_blue - (state['y_offset'] - mtx_rgb[1,2]) * state['z_offset']) / mtx_rgb[1,1]) / 1000
                u_ia_pos = u_ia_pos.reshape(3, 1)

                u_ia_ori = np.array([qw,qx,qy,qz])
                u_ia_ori = u_ia_ori.reshape(4, 1)

                # print("u_ia_pos: ", u_ia_pos)
                visualizer.update_kf(u_ia_ori = u_ia_ori, u_ia_pos = u_ia_pos, gyro = gyro, mag = mag, acc = acc)
                print(f"Posicion c/cam: {float(visualizer.x_estimado[0]):.4f} {float(visualizer.x_estimado[1]):.4f} {float(visualizer.x_estimado[2]):.4f}")
            else:    
                visualizer.update_kf(gyro = gyro, mag = mag, acc = acc)
                # print(f"Posicion s/cam: {float(visualizer.x_estimado[0]):.4f} {float(visualizer.x_estimado[1]):.4f} {float(visualizer.x_estimado[2]):.4f}")

            # graf_queue.put((visualizer.x_estimado[0], visualizer.x_estimado[1], visualizer.x_estimado[2]))

        elif not flag_imu_empty:
            visualizer.update_kf(gyro = gyro, mag = mag, acc = acc)

            # graf_queue.put((visualizer.x_estimado[0], visualizer.x_estimado[1], visualizer.x_estimado[2]))

            # print(f"Posicion s/cam: {float(visualizer.x_estimado[0]):.4f} {float(visualizer.x_estimado[1]):.4f} {float(visualizer.x_estimado[2]):.4f}")

        # end_time = time.time()
        # print(f"Tiempo de procesamiento kalman: {end_time - start_time:.5f} segundos")
        midi_note = map_position_to_midi(float(visualizer.x_estimado[0]), float(visualizer.x_estimado[1]), float(visualizer.x_estimado[2]))
        # print("midi note: ", midi_note)
        send_midi_note(midi_note, acc)
        time.sleep(0.018)  # 18ms entre muestras

with open("detections.txt", "w") as file:
    file.write("Clase, Coordenada_X, Coordenada_Y, Profundidad\n")  # Encabezado del archivo

##########################
# Lanzamos los hilos
##########################
t1 = threading.Thread(target=sensor_capture_thread, daemon=True)
t2 = threading.Thread(target=image_processing_thread, daemon=True)
t3 = threading.Thread(target=kalman_update_thread, daemon=True)

t1.start()
t2.start()
t3.start()

# def update_graf():
#     global contador_grafico
#     X, Y, Z = graf_queue.get()
#     if contador_grafico > 10:
#         update_point(X, Y, Z)
#         contador_grafico = 0
#     else:
#         contador_grafico += 1

# # Temporizador para actualizar la gráfica en tiempo real
# timer = pg.QtCore.QTimer()
# timer.timeout.connect(update_graf)
# timer.start(20)  # Actualiza cada 20 ms 

# Mantenemos el hilo principal vivo (por ejemplo, con un bucle infinito o esperando a que terminen los hilos)
try:
    while not stop_event.is_set():
        # sys.exit(app.exec_())
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminando la ejecución...")
    openni2.unload()
    cv2.destroyAllWindows()
    sys.exit()