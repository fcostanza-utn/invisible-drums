#librerías
from ultralytics import YOLO
import logging
import cv2
from openni import openni2
import math
from collections import deque
import mido
from kalman_module import IMUVisualizer
import sys
import time
import torch
import numpy as np
from data_sync import DataSynchronizer
from queue import Queue, Empty
import threading
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

ref_time_midi_r = 0
ref_time_midi_l = 0

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

# Nombre que se usará para identificar la memoria compartida del graficador.
SHM_NAME_GRAF_RIGHT = 'coords_shared_right'
SHM_NAME_GRAF_LEFT = 'coords_shared_left'
# Nombre que se usará para identificar la memoria compartida del esp.
SHM_NAME_ESP = 'esp_shared'

# Configuración graficos
num_muestras = 100  # Número de muestras visibles en el eje X
x_data = deque(maxlen=num_muestras)
y_data = deque(maxlen=num_muestras)
z_data = deque(maxlen=num_muestras)
t_data = deque(maxlen=num_muestras)  # Contador de muestras (eje X)

"""
Colas de Datos
"""
sensor_queue = Queue(maxsize=100)   # Para datos crudos de la IMU
camara_queue = Queue(maxsize=100)   # Para datos crudos de la cámara

"""
Memoria compartida
"""
# Se crea la memoria compartida para 3 valores de tipo double.
# Shared memory del palillo derecho
shm_graf_r = shared_memory.SharedMemory(create=True, name=SHM_NAME_GRAF_RIGHT, size=3 * np.dtype('d').itemsize)
# Creamos un array numpy que utiliza el buffer de la memoria compartida.
coords_right = np.ndarray((3,), dtype='d', buffer=shm_graf_r.buf)

# Shared memory del palillo izquierdo
shm_graf_l = shared_memory.SharedMemory(create=True, name=SHM_NAME_GRAF_LEFT, size=3 * np.dtype('d').itemsize)
# Creamos un array numpy que utiliza el buffer de la memoria compartida.
coords_left = np.ndarray((3,), dtype='d', buffer=shm_graf_l.buf)

# Intentamos conectar con la memoria compartida ya creada
try:
    shm_esp = shared_memory.SharedMemory(name=SHM_NAME_ESP)
except FileNotFoundError:
    print(f"No se encontró la memoria compartida con nombre '{SHM_NAME_ESP}'. Asegúrate de que el productor la haya creado.")
    exit(1)
imu_data = np.ndarray((1,), dtype='U160', buffer=shm_esp.buf)
"""
INICIALIZACION DE YOLO y kinect
"""
# modelo YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO("yolo-Weights/best_yolo11s_v4Kinect.pt").to(device)
#classNames = ["drumsticks_mid_R", "drumsticks_tip_R", "drumsticks_mid_L", "drumsticks_tip_L"] # Definir las clases de objetos para la detección
classNames = ['3.', 'drumsticks_mid_L', 'drumsticks_mid_R', 'drumsticks_tip_L', 'drumsticks_tip_R']

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
right_kalman  = IMUVisualizer(dt=0.021)
left_kalman  = IMUVisualizer(dt=0.021)
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
    timeoff = 0.01
    if note:
        if 1 < (acc) <= 1.5:
            midi_out.send(mido.Message('note_on', note=note, velocity=15))
            midi_out.send(mido.Message('note_off', note=note, velocity=100))
        elif 1.5 < (acc) <= 2:
            midi_out.send(mido.Message('note_on', note=note, velocity=30))
            midi_out.send(mido.Message('note_off', note=note, velocity=100))
        elif 2 < (acc) <= 2.5:
            midi_out.send(mido.Message('note_on', note=note, velocity=45))
            midi_out.send(mido.Message('note_off', note=note, velocity=100))
        elif 2.5 < (acc) <= 3:
            midi_out.send(mido.Message('note_on', note=note, velocity=60))
            midi_out.send(mido.Message('note_off', note=note, velocity=100))
        elif 3 < (acc) <= 3.5:
            midi_out.send(mido.Message('note_on', note=note, velocity=75))
            midi_out.send(mido.Message('note_off', note=note, velocity=100))
        elif 3.5 < (acc) <= 4:
            midi_out.send(mido.Message('note_on', note=note, velocity=90))
            midi_out.send(mido.Message('note_off', note=note, velocity=100))
        elif 4 < (acc):
            midi_out.send(mido.Message('note_on', note=note, velocity=100))
            midi_out.send(mido.Message('note_off', note=note, velocity=100))

def map_position_to_midi(x, y, z, time_r, time_l, acc_midi):
    global ref_time_midi_r
    global ref_time_midi_l
    x = x * 100
    y = y * 100
    z = z * 100
    #print(f"acc midi: {acc_midi:.2f}")
    print("tiempos midi: ", time_r, time_l)
    if (acc_midi) > 1:
        if time_l - ref_time_midi_l > 0.3:
            if (-30 < x < 0) and (57.5 < y < 72.5) and (-26 < z < 4):           # Nota MIDI para un snare drum
                ref_time_midi_l = time_l
                return 38  
            elif (-65 < x < -35) and (22.5 < y < 37.5) and (-30 < z < 0):       # Nota MIDI para un hihat drum
                ref_time_midi_l = time_l
                return 42  
            elif (-65 < x < -25) and (-10 < y < 5) and (-70 < z < -30):           # Nota MIDI para un crash drum
                ref_time_midi_l = time_l
                return 49  
            elif (11 < x < 51) and (7.5 < y < 22.5) and (-61 < z < -21):       # Nota MIDI para un ride drum
                ref_time_midi_l = time_l
                return 51  
            elif (-30 < x < 0) and (9.5 < y < 24.5) and (-80 < z < -50):       # Nota MIDI para un hightom drum
                ref_time_midi_l = time_l
                return 50  
            elif (17 < x < 47) and (57.5 < y < 72.5) and (-27 < z < 3):        # Nota MIDI para un lowtom drum
                ref_time_midi_l = time_l
                return 45  
        if time_r - ref_time_midi_r > 0.3:
            if (-30 < x < 0) and (57.5 < y < 72.5) and (-26 < z < 4):           # Nota MIDI para un snare drum
                ref_time_midi_r = time_r
                return 38  
            elif (-65 < x < -35) and (22.5 < y < 37.5) and (-30 < z < 0):       # Nota MIDI para un hihat drum
                ref_time_midi_r = time_r
                return 42  
            elif (-65 < x < -25) and (-10 < y < 5) and (-65 < z < -25):           # Nota MIDI para un crash drum
                ref_time_midi_r = time_r
                return 49  
            elif (11 < x < 51) and (7.5 < y < 22.5) and (-61 < z < -21):       # Nota MIDI para un ride drum
                ref_time_midi_r = time_r
                return 51  
            elif (-30 < x < 0) and (9.5 < y < 24.5) and (-80 < z < -50):       # Nota MIDI para un hightom drum
                ref_time_midi_r = time_r
                return 50  
            elif (17 < x < 47) and (57.5 < y < 72.5) and (-27 < z < 3):        # Nota MIDI para un lowtom drum
                ref_time_midi_r = time_r
                return 45  
        return None
    return None

def aprox_depth_disp(depth_frame, Xp, Yp):
    disparidad = 50
    Xp_min = max(Xp - disparidad, 0)
    Xp_max = min(Xp + disparidad, depth_frame.shape[1])  # Asegurar que no exceda las columnas

    # Extraer valores dentro de los límites
    z_value = depth_frame[Yp, Xp_min:Xp_max]
    z_value = z_value[z_value != 0]
    X_match = 0
    z_value_min = 0

    if z_value.size > 0:
        # Obtener el valor mínimo y convertirlo a entero
        z_value_min = int(np.min(z_value))
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
            print("Z value caso 3: ", z_value_min)
    for x in range(Xp_min, Xp_max):
        if depth_frame[Yp, x] == z_value_min: 
            X_match=x
    return {'z_value_min': z_value_min, 'xy':(X_match,Yp)}

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
    while not stop_event.is_set():
        start_time = time.time()
        rwlock.acquire_read()
        try:
            if data_sync.get_state()['button_right'] and data_sync.get_state()['button_left']:
                esp_data = right_kalman.parse_sensor_data(imu_data[0], 0)
                data_sync.set_master(esp_data['pal_indic'])
                data_sync.set_button(esp_data['boton_1'], 'right')
                data_sync.set_button(esp_data['boton_2'], 'left')
                data_sync.update_imu_time(esp_data['timestamp'])
            elif not data_sync.get_state()['button_right'] or not data_sync.get_state()['button_left']:
                if data_sync.get_state()['offset_time_camera'] == 0:
                    data_sync.update_camera_time(time.time())
                esp_data = right_kalman.parse_sensor_data(imu_data[0], 0)
                if esp_data != None:
                    if esp_data.get('boton_1') == 0:
                        data_sync.set_master(esp_data['pal_indic'])
                        data_sync.set_button_repeat(True, 'right')
                        data_sync.update_camera_time(time.time())
                        data_sync.update_imu_time(esp_data['timestamp'])
                    if esp_data.get('boton_2') == 0:
                        data_sync.set_master(esp_data['pal_indic'])
                        data_sync.set_button_repeat(True, 'left')
                        data_sync.update_camera_time(time.time())
                        data_sync.update_imu_time(esp_data['timestamp'])

                    if esp_data['acc_2'] != None:
                        sensor_queue.put((esp_data['acc_1'], esp_data['gyro_1'], esp_data['mag_1'], esp_data['acc_2'], esp_data['gyro_2'], esp_data['mag_2'], esp_data['timestamp']))
                    elif esp_data['acc_2'] == None:
                        sensor_queue.put((esp_data['acc_1'], esp_data['gyro_1'], esp_data['mag_1'], esp_data['pal_indic'] , esp_data['timestamp']))
                    else:
                        pass
                else:
                    pass
        finally:
            rwlock.release_read()
        end_time = time.time()
        #print(f"Tiempo de procesamiento hilo muestreo: {end_time - start_time:.3f} segundos")
        time.sleep(0.005)  # 5ms entre muestras
    #print("Terminando hilo de muestreo...")

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

        depth_buff = np.array(depth_frame.get_buffer_as_uint16()).reshape((depth_frame.height, depth_frame.width))
        mask = (depth_buff == 0).astype(np.uint8) * 255
        depth_corregido = cv2.inpaint(depth_buff, mask, 5, cv2.INPAINT_TELEA)
        #depth_corregido = np.array(depth_frame.get_buffer_as_uint16()).reshape((depth_frame.height, depth_frame.width))

        X_blue = 0
        Y_blue = 0
        Z_blue = 0
        X_green = 0
        Y_green = 0
        Z_green = 0

        if color_data is not None:
            elapsed_time = time.time()
            results = model_yolo.predict(color_data, conf=0.20, stream=True)
            end_time = time.time()
            print(f"Tiempo de procesamiento yolo: {end_time - start_time:.3f} segundos")
            points = {}

            for r in results:
                right_tip_detected = False
                right_mid_detected = False
                left_tip_detected = False
                left_mid_detected = False
                for box in r.boxes:
                    push = False
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    Xp, Yp = (x1 + x2) // 2, (y1 + y2) // 2
                    z_value_dict = aprox_depth_disp(depth_corregido, Xp, Yp)
                    z_value = z_value_dict.get('z_value_min')

                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    if class_name == "drumsticks_tip_R" and right_tip_detected == False:
                        push = True
                        right_tip_detected = True
                    if class_name == "drumsticks_mid_R" and right_mid_detected == False:
                        push = True
                        right_mid_detected = True
                    if class_name == "drumsticks_tip_L" and left_tip_detected == False:
                        push = True
                        left_tip_detected = True                    
                    if class_name == "drumsticks_mid_L" and left_mid_detected == False:
                        push = True
                        left_mid_detected = True
                    if push:
                        cv2.rectangle(color_data, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(color_data, f"z_value:({z_value})",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                        points[class_name] = (Xp, Yp, z_value)

            if right_tip_detected:
                (X_blue, Y_blue, Z_blue) = points["drumsticks_tip_R"]
            if right_mid_detected:
                (X_red, Y_red, Z_red) = points["drumsticks_mid_R"]
            if left_tip_detected:
                (X_green, Y_green, Z_green) = points["drumsticks_tip_L"]
            if left_mid_detected:
                (X_yellow, Y_yellow, Z_yellow) = points["drumsticks_mid_L"]       

            if right_tip_detected or left_tip_detected:
                if ((data_sync.get_state()['button_right'] and data_sync.get_state()['button_left']) or (data_sync.get_state()['button_repeat_right'] or data_sync.get_state()['button_repeat_left'])) and (data_sync.get_state()['master'] == 3 or data_sync.get_state()['master'] == 1):
                    print("Seteando offsets posicion: ", X_blue, Y_blue, Z_blue)
                    data_sync.set_offsets(X_blue, Y_blue, Z_blue, 1000, 1000, 1000)
                    data_sync.set_button_repeat(False, 'right')
                    data_sync.set_button_repeat(False, 'left')
                if ((data_sync.get_state()['button_right'] and data_sync.get_state()['button_left']) or (data_sync.get_state()['button_repeat_right'] or data_sync.get_state()['button_repeat_left'])) and (data_sync.get_state()['master'] == 4 or data_sync.get_state()['master'] == 2):
                    data_sync.set_offsets(1000, 1000, 1000, X_green, Y_green, Z_green)
                    data_sync.set_button_repeat(False, 'right')
                    data_sync.set_button_repeat(False, 'left')

            if right_tip_detected and right_mid_detected:
                if np.linalg.norm(np.array(points["drumsticks_mid_R"][:2]) - np.array(points["drumsticks_tip_R"][:2])) < 250:
                    cv2.line(color_data, points["drumsticks_mid_R"][:2], points["drumsticks_tip_R"][:2], (0, 255, 0), 2)

            if left_tip_detected and left_mid_detected:
                if np.linalg.norm(np.array(points["drumsticks_mid_L"][:2]) - np.array(points["drumsticks_tip_L"][:2])) < 250:
                    cv2.line(color_data, points["drumsticks_mid_L"][:2], points["drumsticks_tip_L"][:2], (0, 255, 0), 2)

            if not data_sync.get_state()['button_right'] or not data_sync.get_state()['button_left']:
                if right_tip_detected and right_mid_detected and left_tip_detected and left_mid_detected:
                    camara_queue.put((X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, X_green, Y_green, Z_green, X_yellow, Y_yellow, Z_yellow, elapsed_time))
                elif right_tip_detected and right_mid_detected and left_tip_detected:
                    camara_queue.put((X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, X_green, Y_green, Z_green, 1000, 1000, 1000, elapsed_time))
                elif right_tip_detected and left_tip_detected and left_mid_detected:
                    camara_queue.put((X_blue, Y_blue, Z_blue, 1000, 1000, 1000, X_green, Y_green, Z_green, X_yellow, Y_yellow, Z_yellow, elapsed_time))
                elif right_tip_detected and left_tip_detected:
                    camara_queue.put((X_blue, Y_blue, Z_blue, 1000, 1000, 1000, X_green, Y_green, Z_green, 1000, 1000, 1000, elapsed_time))
                elif right_tip_detected and right_mid_detected:
                    camara_queue.put((X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, 1000, 1000, 1000, 1000, 1000, 1000, elapsed_time))
                elif left_tip_detected and left_mid_detected:
                    camara_queue.put((1000, 1000, 1000, 1000, 1000, 1000, X_green, Y_green, Z_green, X_yellow, Y_yellow, Z_yellow, elapsed_time))
                elif right_tip_detected:
                    camara_queue.put((X_blue, Y_blue, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, elapsed_time))
                elif left_tip_detected:
                    camara_queue.put((1000, 1000, 1000, 1000, 1000, 1000, X_green, Y_green, Z_green, 1000, 1000, 1000, elapsed_time))
                        
            
            cv2.imshow("Cam", color_data)
            cv2.imshow("Depth", depth_corregido)

        if cv2.waitKey(1) == ord('q'):
            stop_event.set()  # Señalizamos al hilo 1 que debe detenerse
            break
        
    shm_graf_r.close()
    shm_graf_r.unlink()  
    shm_graf_l.close()
    shm_graf_l.unlink()  
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

    u_ia_pos_right = np.zeros((3, 1))
    u_ia_pos_left = np.zeros((3, 1))
    u_ia_ori_right = np.zeros((4, 1))
    u_ia_ori_left = np.zeros((4, 1))
    acc_l = np.zeros((3, 1))
    acc_r = np.zeros((3, 1))
    gyro_r = np.zeros((3, 1))
    gyro_l = np.zeros((3, 1))
    mag_r = np.zeros((3, 1))
    mag_l = np.zeros((3, 1))

    only_blue = False
    only_green = False
    time_only_blue = 0
    time_only_green = 0

    time_note_r = 0
    time_note_l = 0
    
    right_drum_on = True
    left_drum_on = True

    Z_blue = 0
    Z_blue_buff = 0
    Z_red_buff = 0
    Z_green_buff = 0
    Z_yellow_buff = 0

    pal_indic = 0

    q_r = np.zeros((4, 1))
    q_l = np.zeros((4, 1))

    while not stop_event.is_set():
        start_time = time.time()
        try:
            msg = sensor_queue.get(block=False)
            if len(msg) == 7:
                acc_r, gyro_r, mag_r, acc_l, gyro_l, mag_l, milisegundos = msg
                pal_indic = 5
            if len(msg) == 5:
                acc_1, gyro_1, mag_1, pal_indic, milisegundos = msg
                if pal_indic == 3:
                    acc_r = acc_1
                    gyro_r = gyro_1
                    mag_r = mag_1
                if pal_indic == 4:
                    acc_l = acc_1
                    gyro_l = gyro_1
                    mag_l = mag_1
            imu_time = milisegundos - data_sync.get_state()['offset_time_imu']
            flag_imu_empty = False
            flag_no_more_imu = False
        except Empty:
            flag_imu_empty = True
            # print("La cola sensor_queue está vacía.")
        try:
            X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, X_green, Y_green, Z_green, X_yellow, Y_yellow, Z_yellow, elapsed_time = camara_queue.get(block=False)
            camera_time = (elapsed_time - data_sync.get_state()['offset_time_camera'])*1000
            flag_cam_empty = False
            flag_no_more_cam = False
        except Empty:
            flag_cam_empty = True
            # print("La cola camara_queue está vacía.")

        while (camera_time > imu_time) and  not flag_no_more_imu:
            try:
                msg = sensor_queue.get(block=False)
                if len(msg) == 7:
                    acc_r, gyro_r, mag_r, acc_l, gyro_l, mag_l, milisegundos = msg
                    pal_indic = 5
                if len(msg) == 5:
                    acc_1, gyro_1, mag_1, pal_indic, milisegundos = msg
                    #print("pal_indic: ", pal_indic)
                    if pal_indic == 1:
                        acc_r = acc_1
                        gyro_r = gyro_1
                        mag_r = mag_1
                    if pal_indic == 2:
                        acc_l = acc_1
                        gyro_l = gyro_1
                        mag_l = mag_1
                imu_time = milisegundos - data_sync.get_state()['offset_time_imu']
            except Empty:
                flag_no_more_imu = True

        while (((camera_time < imu_time - 50) or (camera_time > imu_time + 50))) and not flag_no_more_cam:
            try:
                X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, X_green, Y_green, Z_green, X_yellow, Y_yellow, Z_yellow, elapsed_time = camara_queue.get(block=False)
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

            if X_yellow == 1000 and Y_yellow == 1000 and Z_yellow == 1000:
                if time_only_green == 0:
                    only_green = True
                    time_only_green = elapsed_time
                elif (elapsed_time - time_only_green) < 100:
                    only_green = True
                else:
                    only_green = False
            else:
                only_green = False
                time_only_green = elapsed_time
            
            if X_blue == 1000 and Y_blue == 1000 and Z_blue == 1000:
                right_drum_on = False
            else:
                right_drum_on = True
                time_ref_right = camera_time
            if X_green == 1000 and Y_green == 1000 and Z_green == 1000:
                left_drum_on = False
            else:
                left_drum_on = True
                time_ref_left = camera_time

            if ((Z_blue_buff + 500) < Z_blue) and Z_blue_buff and right_drum_on and not only_blue:
                Z_blue = Z_blue_buff
                Z_red = Z_red_buff
            else:
                Z_blue_buff = Z_blue
                Z_red_buff = Z_red

            if ((Z_green_buff + 500) < Z_blue) and Z_green_buff and left_drum_on and not only_green:
                Z_green = Z_green_buff
                Z_yellow = Z_yellow_buff
            else:
                Z_green_buff = Z_green
                Z_yellow_buff = Z_yellow

    ################################################# CALCULO DE ORIENTACIÓN MEDIANTE CÁMARA
        if not flag_cam_empty and not only_blue:
            vector_ori = np.array([X_blue - X_red, Y_blue - Y_red, Z_blue - Z_red])
            u = np.array([0,0,1 ])
            
            q_r = look_at(vector_ori,u)
            q_r = q_r/np.linalg.norm(q_r)
        if not flag_cam_empty and not only_green:
            vector_ori = np.array([X_green - X_yellow, Y_green - Y_yellow, Z_green - Z_yellow])
            u = np.array([0,0,1 ])
            
            q_l = look_at(vector_ori,u)
            q_l = q_l/np.linalg.norm(q_l)
    ################################################# ACTUALIZAR FILTROS DE KALMAN     
        state = data_sync.get_state()
        
        if (camera_time > imu_time - 50) and (camera_time < imu_time + 50) and not flag_imu_empty and (pal_indic == 5 or pal_indic == 3):
            if not flag_cam_empty:
                if X_blue and Y_blue and not Z_blue:
                    Z_blue = u_ia_pos_right[2] * 1000 + state['z_offset']
                else:
                    u_ia_pos_right[2] = (Z_blue - state['z_offset']) / 1000
                u_ia_pos_right[0] = (((X_blue - mtx_rgb[0,2]) * Z_blue - (state['x_offset'] - mtx_rgb[0,2]) * state['z_offset']) / mtx_rgb[0,0]) / 1000
                u_ia_pos_right[1] = (((Y_blue - mtx_rgb[1,2]) * Z_blue - (state['y_offset'] - mtx_rgb[1,2]) * state['z_offset']) / mtx_rgb[1,1]) / 1000
                u_ia_pos_right = u_ia_pos_right.reshape(3, 1)

                u_ia_ori_right = q_r
                u_ia_ori_right = u_ia_ori_right.reshape(4, 1)

                right_kalman.update_kf(u_ia_ori = u_ia_ori_right, u_ia_pos = u_ia_pos_right, gyro = gyro_r, mag = mag_r, acc = acc_r)

            else:    
                right_kalman.update_kf(u_ia_ori = u_ia_ori_right, u_ia_pos = u_ia_pos_right, gyro = gyro_r, mag = mag_r, acc = acc_r)
            print(f"Right Kalman both: {float(right_kalman.x_estimado[0]):.2f},{float(right_kalman.x_estimado[1]):.2f},{float(right_kalman.x_estimado[2]):.2f}")
            print("Imu time: ", imu_time)

        elif not flag_imu_empty and (pal_indic == 5 or pal_indic == 3):
            right_kalman.update_kf(gyro = gyro_r, mag = mag_r, acc = acc_r)
            print("Imu time: ", imu_time)
            print("Camera_time : ", camera_time)
                    
        elif not flag_cam_empty and right_drum_on:
            if X_blue and Y_blue and not Z_blue:
                Z_blue = u_ia_pos_right[2] * 1000 + state['z_offset']
            else:
                u_ia_pos_right[2] = (Z_blue - state['z_offset']) / 1000
            u_ia_pos_right[0] = (((X_blue - mtx_rgb[0,2]) * Z_blue - (state['x_offset'] - mtx_rgb[0,2]) * state['z_offset']) / mtx_rgb[0,0]) / 1000
            u_ia_pos_right[1] = (((Y_blue - mtx_rgb[1,2]) * Z_blue - (state['y_offset'] - mtx_rgb[1,2]) * state['z_offset']) / mtx_rgb[1,1]) / 1000
            u_ia_pos_right = u_ia_pos_right.reshape(3, 1)

            u_ia_ori_right = q_r
            u_ia_ori_right = u_ia_ori_right.reshape(4, 1)

            right_kalman.update_kf(u_ia_ori = u_ia_ori_right, u_ia_pos = u_ia_pos_right, gyro = gyro_r, mag = mag_r, acc = acc_r)

        if right_drum_on or (pal_indic == 5 or pal_indic == 3):
            acc_midi = (np.linalg.norm(acc_r) - 1)
            time_note_r = time.time()
            midi_note = map_position_to_midi(float(right_kalman.x_estimado[0].item()), float(right_kalman.x_estimado[1].item()), float(right_kalman.x_estimado[2].item()), time_note_r, -1, acc_midi)
            send_midi_note(midi_note, acc_midi)
            coords_right[0] = float(right_kalman.x_estimado[0].item())
            coords_right[1] = float(right_kalman.x_estimado[1].item())
            coords_right[2] = float(right_kalman.x_estimado[2].item())


        if (camera_time > imu_time - 50) and (camera_time < imu_time + 50) and not flag_imu_empty and (pal_indic == 5 or pal_indic == 4):
            if not flag_cam_empty:
                if X_green and Y_green and not Z_green:
                    Z_green = u_ia_pos_left[2] * 1000 + state['z_offset']
                else:
                    u_ia_pos_left[2] = (Z_green - state['z_offset']) / 1000
                u_ia_pos_left[0] = (((X_green - mtx_rgb[0,2]) * Z_green - (state['x_offset'] - mtx_rgb[0,2]) * state['z_offset']) / mtx_rgb[0,0]) / 1000
                u_ia_pos_left[1] = (((Y_green - mtx_rgb[1,2]) * Z_green - (state['y_offset'] - mtx_rgb[1,2]) * state['z_offset']) / mtx_rgb[1,1]) / 1000
                u_ia_pos_left = u_ia_pos_left.reshape(3, 1)

                u_ia_ori_left = q_l
                u_ia_ori_left = u_ia_ori_left.reshape(4, 1)

                left_kalman.update_kf(u_ia_ori = u_ia_ori_left, u_ia_pos = u_ia_pos_left, gyro = gyro_l, mag = mag_l, acc = acc_l)

            else:    
                left_kalman.update_kf(u_ia_ori = u_ia_ori_left, u_ia_pos = u_ia_pos_left, gyro = gyro_l, mag = mag_l, acc = acc_l)

        elif not flag_imu_empty and (pal_indic == 5 or pal_indic == 4):
            left_kalman.update_kf(gyro = gyro_l, mag = mag_l, acc = acc_l)

        elif not flag_cam_empty and left_drum_on:
            if X_green and Y_green and not Z_green:
                Z_green = u_ia_pos_left[2] * 1000 + state['z_offset']
            else:
                u_ia_pos_left[2] = (Z_green - state['z_offset']) / 1000
            u_ia_pos_left[0] = (((X_green - mtx_rgb[0,2]) * Z_green - (state['x_offset'] - mtx_rgb[0,2]) * state['z_offset']) / mtx_rgb[0,0]) / 1000
            u_ia_pos_left[1] = (((Y_green - mtx_rgb[1,2]) * Z_green - (state['y_offset'] - mtx_rgb[1,2]) * state['z_offset']) / mtx_rgb[1,1]) / 1000
            u_ia_pos_left = u_ia_pos_left.reshape(3, 1)

            u_ia_ori_left = q_l
            u_ia_ori_left = u_ia_ori_left.reshape(4, 1)

            left_kalman.update_kf(u_ia_ori = u_ia_ori_left, u_ia_pos = u_ia_pos_left, gyro = gyro_l, mag = mag_l, acc = acc_l)

        if left_drum_on or (pal_indic == 5 or pal_indic == 4):
            acc_midi = (np.linalg.norm(acc_l) - 1)
            time_note_l = time.time()
            midi_note = map_position_to_midi(float(left_kalman.x_estimado[0].item()), float(left_kalman.x_estimado[1].item()), float(left_kalman.x_estimado[2].item()), -1, time_note_l, acc_midi)
            send_midi_note(midi_note, acc_midi)
            coords_left[0] = float(left_kalman.x_estimado[0].item())
            coords_left[1] = float(left_kalman.x_estimado[1].item())
            coords_left[2] = float(left_kalman.x_estimado[2].item())

        end_time = time.time()
        #print(f"Tiempo de procesamiento hilo kalman: {end_time - start_time:.3f} segundos")
        time.sleep(0.006)  # 8ms entre muestras

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
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminando la ejecución...")
    openni2.unload()
    cv2.destroyAllWindows()
    shm_graf_r.close()
    shm_graf_l.close()
    shm_esp.close()
    shm_graf_r.unlink()
    shm_graf_l.unlink()
    sys.exit()