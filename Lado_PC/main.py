#librerías
from ultralytics import YOLO
import logging
import cv2
from openni import openni2
import math
from collections import deque
import mido
from kalman_module import IMUVisualizer
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
import csv
from multiprocessing import shared_memory
from RW_LOCK import RWLock

"""
Variables de datos y control
"""
init_meseaurement = False
processing_flag = True

data_sync = DataSynchronizer()      # Instancia compartida para sincronizar datos entre hilos

stop_event = threading.Event()      # Creamos un evento que actuará como bandera para detener el hilo

rwlock = RWLock()

tip_points = []
mid_points = []

blue_tip_detection = None
red_tape_detection = None

ref_time_midi = 0

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

contador_imu = 0

# Nombre que se usará para identificar la memoria compartida del graficador.
SHM_NAME_GRAF = 'coords_shared'
# Nombre que se usará para identificar la memoria compartida del esp.
SHM_NAME_ESP = 'esp_shared'

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
Memoria compartida
"""
# Se crea la memoria compartida para 3 valores de tipo double.
# El tamaño se calcula como 3 * tamaño de un double.
shm_graf = shared_memory.SharedMemory(create=True, name=SHM_NAME_GRAF, size=3 * np.dtype('d').itemsize)
# Creamos un array numpy que utiliza el buffer de la memoria compartida.
coords = np.ndarray((3,), dtype='d', buffer=shm_graf.buf)

# Intentamos conectar con la memoria compartida ya creada
try:
    shm_esp = shared_memory.SharedMemory(name=SHM_NAME_ESP)
except FileNotFoundError:
    print(f"No se encontró la memoria compartida con nombre '{SHM_NAME_ESP}'. Asegúrate de que el productor la haya creado.")
    exit(1)
imu_data = np.ndarray((1,), dtype='U120', buffer=shm_esp.buf)
"""
INICIALIZACION DE YOLO y Midas
"""
# modelo YOLO
# modelo YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO("yolo-Weights/best_yolo11m_v3Kinect.pt").to(device)
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
visualizer = IMUVisualizer(dt=0.02)
"""
INICIALIZACION DE MIDI
"""
print("Available MIDI output ports:")
print(mido.get_output_names())
portmidi = mido.Backend('mido.backends.rtmidi')
midi_out = portmidi.open_output('MIDI1 1')

"""
FUNCIONES
"""
# Función para enviar una nota MIDI
def send_midi_note(note, acc):
    timeoff = 0.05
    if note:
        if 0.5 < (acc) <= 1:
            midi_out.send(mido.Message('note_on', note=note, velocity=15))
            midi_out.send(mido.Message('note_off', note=note, velocity=100, time=timeoff))
        elif 1 < (acc) <= 1.5:
            midi_out.send(mido.Message('note_on', note=note, velocity=30))
            midi_out.send(mido.Message('note_off', note=note, velocity=100, time=timeoff))
        elif 1.5 < (acc) <= 2:
            midi_out.send(mido.Message('note_on', note=note, velocity=45))
            midi_out.send(mido.Message('note_off', note=note, velocity=100, time=timeoff))
        elif 2 < (acc) <= 2.5:
            midi_out.send(mido.Message('note_on', note=note, velocity=60))
            midi_out.send(mido.Message('note_off', note=note, velocity=100, time=timeoff))
        elif 2.5 < (acc) <= 3:
            midi_out.send(mido.Message('note_on', note=note, velocity=75))
            midi_out.send(mido.Message('note_off', note=note, velocity=100, time=timeoff))
        elif 3 < (acc) <= 3.5:
            midi_out.send(mido.Message('note_on', note=note, velocity=90))
            midi_out.send(mido.Message('note_off', note=note, velocity=100, time=timeoff))
        elif 3.5 < (acc):
            midi_out.send(mido.Message('note_on', note=note, velocity=100))
            midi_out.send(mido.Message('note_off', note=note, velocity=100, time=timeoff))

def map_position_to_midi(x, y, z, time, acc_midi):
    global ref_time_midi
    x = x * 100
    y = y * 100
    z = z * 100
    #print(f"acc midi: {acc_midi:.2f}")
    if (acc_midi) > 0.25:
        if time - ref_time_midi > 0.25:
            if (-24 < x < 6) and (45 < y < 55) and (-26 < z < 4):           # Nota MIDI para un snare drum
                ref_time_midi = time
                return 38  
            elif (-56 < x < -26) and (25 < y < 35) and (-24 < z < 6):       # Nota MIDI para un hihat drum
                ref_time_midi = time
                return 42  
            elif (-53 < x < -13) and (-2.5 < y < 7.5) and (-56 < z < -16):           # Nota MIDI para un crash drum
                ref_time_midi = time
                return 49  
            elif (11 < x < 51) and (15 < y < 25) and (-61 < z < -21):       # Nota MIDI para un ride drum
                ref_time_midi = time
                return 51  
            elif (-27 < x < 3) and (12.0 < y < 22.0) and (-67 < z < -37):       # Nota MIDI para un hightom drum
                ref_time_midi = time
                return 50  
            elif (17 < x < 47) and (45 < y < 55) and (-27 < z < 3):        # Nota MIDI para un lowtom drum
                ref_time_midi = time
                return 45  
        return None
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
            disparidad = 125
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

def guardar_en_csv(nombre_archivo, dato1, dato2, dato3, dato4):
    with open(nombre_archivo, mode='a', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([dato1, dato2, dato3, dato4])
    # print(f"Datos guardados en {nombre_archivo}")

def quaternion_from_matrix(M):
    """
    Convierte una matriz de rotación 3x3 en un cuaternión.
    La convención del cuaternión es [qw, qx, qy, qz].
    """
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (M[2, 1] - M[1, 2]) / S
        qy = (M[0, 2] - M[2, 0]) / S
        qz = (M[1, 0] - M[0, 1]) / S
    elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
        S = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2  # S = 4 * qx
        qw = (M[2, 1] - M[1, 2]) / S
        qx = 0.25 * S
        qy = (M[0, 1] + M[1, 0]) / S
        qz = (M[0, 2] + M[2, 0]) / S
    elif M[1, 1] > M[2, 2]:
        S = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2  # S = 4 * qy
        qw = (M[0, 2] - M[2, 0]) / S
        qx = (M[0, 1] + M[1, 0]) / S
        qy = 0.25 * S
        qz = (M[1, 2] + M[2, 1]) / S
    else:
        S = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2  # S = 4 * qz
        qw = (M[1, 0] - M[0, 1]) / S
        qx = (M[0, 2] + M[2, 0]) / S
        qy = (M[1, 2] + M[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def look_at(forward, up=np.array([0, 0, 1])):
    """
    Construye un cuaternión a partir de:
      - forward: vector de dirección hacia donde se quiere orientar.
      - up: vector que indica el "arriba" en el mundo.
    Se calcula un sistema de ejes ortonormal en el que:
      - f es el eje forward (dirección deseada),
      - r es el eje right (perpendicular a forward y up),
      - u es el eje up recalculado para garantizar ortogonalidad.
    La matriz de rotación se construye como:
         [ r_x   u_x   f_x ]
         [ r_y   u_y   f_y ]
         [ r_z   u_z   f_z ]
    y se convierte a cuaternión.
    """
    # Normalizar el vector forward
    norm_forward = np.linalg.norm(forward)
    f = forward / norm_forward

    # Verificar que el vector up no sea colineal con forward.
    if abs(np.dot(f, up)) > 0.999:
        # Si lo es, se elige otro vector up arbitrario.
        up = np.array([0, 1, 0])

    # Calcular el vector right (r = f x up)
    r = np.cross(f, up)
    norm_r = np.linalg.norm(r)
    r /= norm_r

    # Recalcular up para que sea ortonormal (u = r x f)
    u = np.cross(r, f)

    # Construir la matriz de rotación
    # Nota: Esta matriz rota un vector del sistema local al sistema global.
    M = np.array([
        [r[0], u[0], f[0]],
        [r[1], u[1], f[1]],
        [r[2], u[2], f[2]]
    ])

    # Convertir la matriz a cuaternión
    q = quaternion_from_matrix(M)
    return q


##########################
# 1. Hilo de captura (sensor)
##########################
def sensor_capture_thread():
    global contador_imu
    while not stop_event.is_set():
        start_time = time.time()
        rwlock.acquire_read()
        try:
            if data_sync.get_state()['button']:
                #print("imu_data: ",imu_data)
                _, _, _, ref_time, ref_but = visualizer.parse_sensor_data(imu_data[0], 0)
                data_sync.set_button(ref_but)
                data_sync.update_imu_time(ref_time)
            if not data_sync.get_state()['button']:
                if data_sync.get_state()['offset_time_camera'] == 0:
                    data_sync.update_camera_time(time.time())
                #print("imu_data: ",imu_data)
                acc, gyro, mag, ref_time, ref_but = visualizer.parse_sensor_data(imu_data[0], 0)
                if ref_but == 0:
                    data_sync.set_button_repeat(True)
                    data_sync.update_camera_time(time.time())
                    data_sync.update_imu_time(ref_time)
                    ref_but = 1
                sensor_queue.put((acc, gyro, mag, ref_time))
                end_time = time.time()
                # print(f"Tiempo de procesamiento captura: {end_time - start_time:.5f} segundos")
                contador_imu += 1
        finally:
            rwlock.release_read()
        time.sleep(0.005)  # 5ms entre muestras

    
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
            results = model_yolo.predict(color_data, conf=0.25, stream=True)
            points = {}

            for r in results:
                # depth_corregido = corregir_mapa_profundidad(depth_data, T_prof, mtx_prof[0,0], mtx_prof[1,1], mtx_prof[0,2], mtx_prof[1,2])
                tip_detected = False
                mid_detected = False
                for box in r.boxes:
                    push = False
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

                    if class_name == "drumsticks_tip" and tip_detected == False:
                        push = True
                        tip_detected = True
                    
                    if class_name == "drumsticks_mid" and mid_detected == False:
                        push = True
                        mid_detected = True
                    
                    if push:
                        cv2.rectangle(color_data, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(color_data, f"{class_name} {confidence} Coord:({Xp},{Yp},{z_value})",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        points[class_name] = (Xp, Yp, z_value)

            # Dibujar línea entre drumsticks_mid y drumsticks_tip
            if "drumsticks_mid" in points and "drumsticks_tip" in points:
                if np.linalg.norm(np.array(points["drumsticks_mid"][:2]) - np.array(points["drumsticks_tip"][:2])) < 250:
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
            elif "drumsticks_tip" in points:
                (X_blue, Y_blue, Z_blue) = points["drumsticks_tip"]
                if data_sync.get_state()['button']:
                    data_sync.set_offsets(X_blue, Y_blue, Z_blue)
                if data_sync.get_state()['button_repeat']:
                    data_sync.set_offsets(X_blue, Y_blue, Z_blue)
                    data_sync.set_button_repeat(False)  
                if not data_sync.get_state()['button']:
                    camara_queue.put((X_blue, Y_blue, Z_blue, 1000, 1000, 1000, elapsed_time))

            cv2.imshow("Cam", color_data)
            cv2.imshow("Depth", depth_corregido)

        end_time = time.time()
        # print(f"Tiempo de procesamiento camara: {end_time - start_time:.3f} segundos")
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()  # Señalizamos al hilo 1 que debe detenerse
            break
        time.sleep(0.005)  # 5ms entre muestras
    
    shm_graf.close()
    shm_graf.unlink()  
    shm_esp.close() 
    openni2.unload()
    cv2.destroyAllWindows()

##########################
# 3. Hilo de actualización del Kalman
##########################
def kalman_update_thread():
    global contador_imu
    flag_imu_empty = False
    flag_cam_empty = False
    flag_no_more_cam = False
    flag_no_more_imu = False
    camera_time = 0
    imu_time = 0
    u_ia_pos = np.zeros((3, 1))
    u_ia_ori = np.zeros((4, 1))
    acc = np.zeros((3, 1))
    note = None
    only_blue = False
    time_only_blue = 0
    qw = 0
    qx = 0
    qy = 0
    qz = 0
    kalman_time = 0
    
    Z_blue_buff = 0
    Z_red_buff = 0
    open("datos.csv", mode='w', newline='')

    while not stop_event.is_set():
        start_time = time.time()
        try:
            acc, gyro, mag, milisegundos = sensor_queue.get(block=False)
            imu_time = milisegundos - data_sync.get_state()['offset_time_imu']
            flag_imu_empty = False
            flag_no_more_imu = False
        except Empty:
            flag_imu_empty = True
            #print("La cola sensor_queue está vacía.")
        try:
            X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, elapsed_time = camara_queue.get(block=False)
            camera_time = (elapsed_time - data_sync.get_state()['offset_time_camera'])*1000
            flag_cam_empty = False
            flag_no_more_cam = False
        except Empty:
            flag_cam_empty = True
            # print("La cola camara_queue está vacía.")

        while (camera_time > imu_time) and  not flag_no_more_imu:
            try:
                acc, gyro, mag, milisegundos = sensor_queue.get(block=False)
                imu_time = milisegundos - data_sync.get_state()['offset_time_imu']
            except Empty:
                flag_no_more_imu = True  

        while (((camera_time < imu_time - 50) or (camera_time > imu_time + 50))) and not flag_no_more_cam:
            try:
                X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, elapsed_time = camara_queue.get(block=False)
                camera_time = (elapsed_time - data_sync.get_state()['offset_time_camera'])*1000
            except Empty:
                flag_no_more_cam = True

        if not flag_cam_empty:
            if X_red == 1000 and Y_red == 1000 and Z_red == 1000:
                if time_only_blue == 0:
                    only_blue = True
                    time_only_blue = elapsed_time
                elif (elapsed_time - time_only_blue) < 100:
                    only_blue = True
                else:
                    only_blue = False
            else:
                only_blue = False
                time_only_blue = elapsed_time

            if ((Z_blue_buff + 500) < Z_blue) and Z_blue_buff:
                Z_blue = Z_blue_buff
                Z_red = Z_red_buff
            else:
                Z_blue_buff = Z_blue
                Z_red_buff = Z_red
    ################################################# CALCULO DE ORIENTACIÓN MEDIANTE CÁMARA
        if not flag_cam_empty and not only_blue:
            vector_ori = np.array([X_blue - X_red, Y_blue - Y_red, Z_red - Z_blue])
            u = np.array([0,0,1 ])
            
            q = look_at(vector_ori,u)
            q = q/np.linalg.norm(q)
            print("q: ", q)
    ################################################# ACTUALIZAR FILTROS DE KALMAN     
        # print(f"Tiempo de la cámara: {camera_time:.0f} ms")
        # print(f"Tiempo del IMU: {imu_time:.0f} ms")
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

                u_ia_ori = q
                u_ia_ori = u_ia_ori.reshape(4, 1)

                # print("u_ia_pos: ", u_ia_pos)
                visualizer.update_kf(u_ia_ori = u_ia_ori, u_ia_pos = u_ia_pos, gyro = gyro, mag = mag, acc = acc)
                # print(f"Posicion c/cam: {float(visualizer.x_estimado[0]):.4f} {float(visualizer.x_estimado[1]):.4f} {float(visualizer.x_estimado[2]):.4f}")
            else:    
                visualizer.update_kf(u_ia_ori = u_ia_ori, u_ia_pos = u_ia_pos, gyro = gyro, mag = mag, acc = acc)
                # print(f"Posicion s/cam: {float(visualizer.x_estimado[0]):.4f} {float(visualizer.x_estimado[1]):.4f} {float(visualizer.x_estimado[2]):.4f}")

            # graf_queue.put((visualizer.x_estimado[0], visualizer.x_estimado[1], visualizer.x_estimado[2]))

        elif not flag_imu_empty:
            visualizer.update_kf(gyro = gyro, mag = mag, acc = acc)

            # graf_queue.put((visualizer.x_estimado[0], visualizer.x_estimado[1], visualizer.x_estimado[2]))

            # print(f"Posicion s/cam: {float(visualizer.x_estimado[0]):.4f} {float(visualizer.x_estimado[1]):.4f} {float(visualizer.x_estimado[2]):.4f}")
        
        elif not flag_cam_empty:
            if X_blue and Y_blue and not Z_blue:
                Z_blue = u_ia_pos[2] * 1000 + state['z_offset']
            else:
                u_ia_pos[2] = (Z_blue - state['z_offset']) / 1000
            u_ia_pos[0] = (((X_blue - mtx_rgb[0,2]) * Z_blue - (state['x_offset'] - mtx_rgb[0,2]) * state['z_offset']) / mtx_rgb[0,0]) / 1000
            u_ia_pos[1] = (((Y_blue - mtx_rgb[1,2]) * Z_blue - (state['y_offset'] - mtx_rgb[1,2]) * state['z_offset']) / mtx_rgb[1,1]) / 1000
            u_ia_pos = u_ia_pos.reshape(3, 1)

            u_ia_ori = q
            u_ia_ori = u_ia_ori.reshape(4, 1)

            # print("u_ia_pos: ", u_ia_pos)
            visualizer.update_kf(u_ia_ori = u_ia_ori, u_ia_pos = u_ia_pos, gyro = gyro, mag = mag, acc = acc)
            #print(f"Posicion c/cam: {float(visualizer.x_estimado[0]):.4f} {float(visualizer.x_estimado[1]):.4f} {float(visualizer.x_estimado[2]):.4f}")
        
        #print("Velocidad: ", float(visualizer.x_estimado[3]),float(visualizer.x_estimado[4]),float(visualizer.x_estimado[5]))
        #print("u_ia_ori: ", u_ia_ori)
        acc_midi = (np.linalg.norm(acc) - 1)
        midi_note = map_position_to_midi(float(visualizer.x_estimado[0]), float(visualizer.x_estimado[1]), float(visualizer.x_estimado[2]), time.time(), acc_midi)
        send_midi_note(midi_note, acc_midi)
        coords[0] = float(visualizer.x_estimado[0])
        coords[1] = float(visualizer.x_estimado[1])
        coords[2] = float(visualizer.x_estimado[2])


        end_time = time.time()
        time_kalman = end_time - start_time
        guardar_en_csv("datos.csv", camera_time, imu_time, time_kalman, contador_imu)
        # print(f"Tiempo de procesamiento kalman: {end_time - start_time:.5f} segundos")
        time.sleep(0.01)  # 10ms entre muestras

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


# Mantenemos el hilo principal vivo (por ejemplo, con un bucle infinito o esperando a que terminen los hilos)
try:
    while not stop_event.is_set():
        # sys.exit(app.exec_())
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminando la ejecución...")
    openni2.unload()
    cv2.destroyAllWindows()
    shm_graf.close()
    shm_esp.close()
    shm_graf.unlink()
    sys.exit()