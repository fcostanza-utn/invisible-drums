#librerías
from ultralytics import YOLO
import cv2
from openni import openni2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mido
from imu_module import IMUVisualizer
from midas import DepthEstimator
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import sys
import time
import torch
import numpy as np
from Coordinate3DCalculator import Coordinate3DCalculator
"""
Variables de datos y control
"""
init_meseaurement = False
processing_flag = True
"""
INICIALIZACION DE YOLO
"""
# modelo YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO("yolo-Weights/best_yolo11m_v2.pt").to(device)
classNames = ["drumsticks_mid", "drumsticks_tip"] # Definir las clases de objetos para la detección
# Midas
#depth_estimator = DepthEstimator()
#Cal_Poly = np.poly1d([-8.94471124, 11.59243574, -1.75101498])

# Iniciar la cámara
openni2.initialize("C:/Program Files/OpenNI2/Redist")
mode = openni2.VideoMode(
        pixelFormat=openni2.PIXEL_FORMAT_RGB888,  #formato de 24 bits
        resolutionX=640,
        resolutionY=480,
        fps=60
    )
device_ni = openni2.Device.open_any()
depth_stream = device_ni.create_depth_stream()
color_stream = device_ni.create_color_stream()
color_stream.set_video_mode(mode)
depth_stream.start()
color_stream.start()
"""cap = cv2.VideoCapture(0, cv2.CAP_OPENNI2)
if not cap.isOpened():
    print("Error: No se pudo abrir el dispositivo OpenNI2.")
    sys.exit()
cap.set(cv2.CAP_PROP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv2.CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE_BGR_IMAGE)
cap.set(3, 640)  # Ancho del fotograma
cap.set(4, 480)  # Alto del fotograma"""



tip_points = []
mid_points = []

blue_tip_detection = None
red_tape_detection = None

"""
INICIALIZACION DE IMU
"""
visualizer = IMUVisualizer('192.168.1.71', 80, 0.02)

"""
INICIALIZACION DE MIDI
"""
# Función para enviar una nota MIDI
def send_midi_note(note):
    if note:
        midi_out.send(mido.Message('note_on', note=note, velocity=100))
        midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve

print("Available MIDI output ports:")
print(mido.get_output_names())

portmidi = mido.Backend('mido.backends.rtmidi')
midi_out = portmidi.open_output('MIDI1 1')

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

"""
Clase Hilo para capturar datos
"""
class SensorCaptureThread(QThread):
    dataReady = pyqtSignal(object, object, object)  # Señal para enviar datos al hilo principal

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        global processing_flag
        button = True
        while self.running:
            imu_data = visualizer.receive_data()  # Leer IMU
            if button:
                _, _, _, _, button = visualizer.parse_sensor_data(imu_data, 0)
            color_frame = color_stream.read_frame()
            depth_frame = depth_stream.read_frame()
            color_data = np.array(color_frame.get_buffer_as_triplet()).reshape((color_frame.height, color_frame.width, 3))    
            color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
            depth_data = np.array(depth_frame.get_buffer_as_uint16()).reshape((depth_frame.height, depth_frame.width))

            if color_data is not None and imu_data and processing_flag and (button == False):
                processing_flag = False
                self.dataReady.emit(imu_data, color_data, depth_data)
            time.sleep(0.015)  # Esperar 60ms para el siguiente frame

    def stop(self):
        self.running = False
        self.wait()
"""
Funcion process_data: procesa los datos de la IMU y la cámara
"""
def process_data(imu_data, color_img, depth_img):
    global processing_flag
    #print("IMU data: ", imu_data)
    start_time = time.time()
    image_height = color_img.shape[0]
    #depth_map = depth_estimator.estimate_depth(img)
    #depth_map = depth_estimator.ConvertToAbsoluteDepth(depth_estimator.estimate_depth(img),Cal_Poly)
    results = model_yolo.predict(color_img, conf=0.35, stream=True)
    """calculator = Coordinate3DCalculator(
        focal_length_mm=7.9,        # Distancia focal de tu cámara
        sensor_width_mm=9.40741,      # Ancho del sensor de tu cámara
        image_width_pixels=640,   # Resolución horizontal de tu cámara
        known_distance_cm=20       # Distancia medida entre punta y cinta
    )"""
    points = {}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            Xp, Yp = (x1 + x2) // 2, (y1 + y2) // 2
            z_value = depth_img[Yp, Xp]

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            cv2.rectangle(color_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(color_img, f"{class_name} {confidence} Coord:({Xp},{Yp},{z_value:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            points[class_name] = (Xp, Yp)

    # Dibujar línea entre drumsticks_mid y drumsticks_tip
    if "drumsticks_mid" in points and "drumsticks_tip" in points:
        cv2.line(color_img, points["drumsticks_mid"], points["drumsticks_tip"], (0, 255, 0), 2)
        (X, Y) = points["drumsticks_tip"]
        """
        fix_position=visualizer.update(raw_data=imu_data)
        print("fix_position: ", fix_position)
        blue_tip_detection = points["drumsticks_tip"]    
        red_tape_detection = points["drumsticks_mid"]    
        
        # Calcular coordenadas 3D
        x, y, z = calculator.calculate_coordinates(
            blue_tip_detection, 
            red_tape_detection,
            image_height
        )
        
        # Obtener métricas de confianza
        confidence = calculator.get_confidence_metrics(
            blue_tip_detection,
            red_tape_detection
        )
        
        print(f"Coordenadas de la punta:")
        print(f"X: {x:.2f} cm")
        print(f"Y: {y:.2f} cm")
        print(f"Z: {z:.2f} cm")
        print(f"\nMétricas de confianza:")
        print(f"Distancia en píxeles: {confidence['pixel_distance']:.2f}")
        print(f"Incertidumbre relativa: {confidence['relative_depth_uncertainty']:.2f}")
        """
        midi_note = map_position_to_midi(X, Y)
        send_midi_note(midi_note)

    cv2.imshow("Cam", color_img)
    cv2.imshow("Depth", depth_img)

    if cv2.waitKey(1) == ord('q'):
        sensor_thread.stop()
        openni2.unload()
        cv2.destroyAllWindows()
        sys.exit()
    processing_flag = True
    end_time = time.time()
    print(f"Tiempo de procesamiento: {end_time - start_time:.3f} segundos")

    
with open("detections.txt", "w") as file:
    file.write("Clase, Coordenada_X, Coordenada_Y, Profundidad\n")  # Encabezado del archivo

sensor_thread = SensorCaptureThread()
sensor_thread.dataReady.connect(process_data)
sensor_thread.start()

#visualizer.view.show()
visualizer.app.exec_()

"""
# Graficar el movimiento en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar puntos de "drumsticks_tip" en rojo y "drumsticks_mid" en azul
if tip_points:
    X_tip, Y_tip, Z_tip = zip(*tip_points)
    ax.plot(X_tip, Y_tip, Z_tip, color='red', label="drumsticks_tip")
if mid_points:
    X_mid, Y_mid, Z_mid = zip(*mid_points)
    ax.plot(X_mid, Y_mid, Z_mid, color='blue', label="drumsticks_mid")

# Etiquetas y visualización
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth (Z)')
ax.legend()
plt.show()
"""