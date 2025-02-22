#librerías
from ultralytics import YOLO
import cv2
from openni import openni2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mido
from imu_module import IMUVisualizer
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import sys
import time
import torch
import numpy as np
from Coordinate3DCalculator import Coordinate3DCalculator
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

"""
Colas de Datos
"""
sensor_queue = Queue(maxsize=100)   # Para datos crudos de la IMU
camara_queue = Queue(maxsize=100)   # Para datos crudos de la cámara

"""
INICIALIZACION DE YOLO y Midas
"""
# modelo YOLO
# modelo YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO("yolo-Weights/best_yolo11m_vKinect.pt").to(device)
classNames = ["drumsticks_mid", "drumsticks_tip"] # Definir las clases de objetos para la detección

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
#depth_stream.set_mirroring_enabled(False)
#color_stream.set_mirroring_enabled(False)
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

"""
FUNCIONES
"""
# Función para enviar una nota MIDI
def send_midi_note(note):
    if note:
        midi_out.send(mido.Message('note_on', note=note, velocity=100))
        midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve

def map_position_to_midi(x, y):
    if x < 100 and y < 100:
        return 36  # Nota MIDI para un kick drum
    elif x < 200 and y < 200:
        return 38  # Nota MIDI para un snare drum
    elif x < 200 and y < 200: 
        return 42  # Nota MIDI para un hihat drum
    elif x < 200 and y < 200:
        return 49  # Nota MIDI para un crash drum
    elif x < 200 and y < 200:
        return 51  # Nota MIDI para un ride drum
    elif x < 200 and y < 200:
        return 50  # Nota MIDI para un hightom drum
    elif x < 200 and y < 200:
        return 45  # Nota MIDI para un lowtom drum
    return None
    
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
                sensor_queue.put((imu_data))
                end_time = time.time()
                # print(f"Tiempo de procesamiento captura: {end_time - start_time:.5f} segundos")
            time.sleep(0.01)  # 10ms entre muestras

    
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
        depth_data = np.array(depth_frame.get_buffer_as_uint16()).reshape((depth_frame.height, depth_frame.width))

        if color_data is not None:
            elapsed_time = time.time()
            image_height = color_data.shape[0]
            #depth_map = depth_estimator.ConvertToAbsoluteDepth(depth_estimator.estimate_depth(img),Cal_Poly)
            results = model_yolo.predict(color_data, conf=0.35, stream=True)
            points = {}

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    Xp, Yp = (x1 + x2) // 2, (y1 + y2) // 2
                    z_value = int(depth_data[Yp, Xp])

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
                if not data_sync.get_state()['button']:
                    camara_queue.put((X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, elapsed_time))

            cv2.imshow("Cam", color_data)
            cv2.imshow("Depth", depth_data)

        end_time = time.time()
        # print(f"Tiempo de procesamiento camara: {end_time - start_time:.3f} segundos")
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()  # Señalizamos al hilo 1 que debe detenerse
            break
        time.sleep(0.001)  # 5ms entre muestras
    
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

    while not stop_event.is_set():
        start_time = time.time()
        try:
            imu_data = sensor_queue.get(block=False)
            flag_imu_empty = False
        except Empty:
            flag_imu_empty = True
            print("La cola sensor_queue está vacía.")
        try:
            X_blue, Y_blue, Z_blue, X_red, Y_red, Z_red, elapsed_time = camara_queue.get(block=False)
            flag_cam_empty = False
        except Empty:
            flag_cam_empty = True
            print("La cola camara_queue está vacía.")

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
            acc, gyro, mag, milisegundos, _ = visualizer.parse_sensor_data(data_string = imu_data, ref = 0)
            imu_time = milisegundos - data_sync.get_state()['offset_time_imu']
            print("imu_time: ", imu_time)
        
        if not flag_cam_empty:   
            camera_time = (elapsed_time - data_sync.get_state()['offset_time_camera'])*1000
            print("camera_time: ", camera_time)

        if (camera_time > imu_time - 20) and (camera_time < imu_time + 20) and not flag_imu_empty and not flag_cam_empty:
            state = data_sync.get_state()
            print("X_blue: ", X_blue)
            u_ia_pos = np.array([X_blue - state['x_offset'], Y_blue - state['y_offset'], Z_blue - state['z_offset']])
            u_ia_pos = u_ia_pos.reshape(3, 1)
            u_ia_ori = np.array([qw,qx,qy,qz])
            u_ia_ori = u_ia_ori.reshape(4, 1)

            u_ia_pos[2] = u_ia_pos[2] / 1000
            u_ia_pos[0] = (u_ia_pos[0] - 330) * u_ia_pos[2] / 600
            u_ia_pos[1] = (u_ia_pos[1] - 240) * u_ia_pos[2] / 600
            print("Z_blue: ", Z_blue)
            print("z_offset: ", state['z_offset'])
            print("u_ia_pos: ", u_ia_pos)
            visualizer.update_kf(u_ia_ori = u_ia_ori, u_ia_pos = u_ia_pos, gyro = gyro, mag = mag, acc = acc)
            print("fix_position (c/cam): ", visualizer.x_estimado)
        elif not flag_imu_empty:
            visualizer.update_kf(gyro = gyro, mag = mag, acc = acc)
            print("fix_position (s/cam): ", visualizer.x_estimado)

        end_time = time.time()
        # print(f"Tiempo de procesamiento kalman: {end_time - start_time:.5f} segundos")
        
        # midi_note = map_position_to_midi(X_blue, Y_blue)
        # send_midi_note(midi_note)
            
        time.sleep(0.005)  # 5ms entre muestras


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
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminando la ejecución...")
    openni2.unload()
    cv2.destroyAllWindows()
    sys.exit()