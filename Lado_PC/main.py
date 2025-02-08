#librerías
from ultralytics import YOLO
import cv2
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
"""
Variables de datos y control
"""
init_meseaurement = False
processing_flag = True
"""
INICIALIZACION DE YOLO y Midas
"""
# modelo YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO("yolo-Weights/best_yolo11m_v2.pt").to(device)
classNames = ["drumsticks_mid", "drumsticks_tip"] # Definir las clases de objetos para la detección
# Midas
depth_estimator = DepthEstimator()
Cal_Poly = np.poly1d([-8.94471124, 11.59243574, -1.75101498])
# Iniciar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)  # Ancho del fotograma
cap.set(4, 480)  # Alto del fotograma

tip_points = []
mid_points = []

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
    dataReady = pyqtSignal(object, object)  # Señal para enviar datos al hilo principal

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
            success, img = cap.read()  # Capturar frame de cámara
            if success and imu_data and processing_flag and button==False:
                processing_flag = False
                self.dataReady.emit(imu_data, img)  # Emitir datos al hilo principal

            time.sleep(0.015)  # Esperar 60ms para el siguiente frame

    def stop(self):
        self.running = False
        self.wait()
"""
Funcion process_data: procesa los datos de la IMU y la cámara
"""
def process_data(imu_data, img):
    global processing_flag
    #print("IMU data: ", imu_data)
    start_time = time.time()
    depth_map = depth_estimator.estimate_depth(img)
    #depth_map = depth_estimator.ConvertToAbsoluteDepth(depth_estimator.estimate_depth(img),Cal_Poly)
    results = model_yolo.predict(img, conf=0.6, stream=True)
    points = {}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            Xp, Yp = (x1 + x2) // 2, (y1 + y2) // 2
            z_value = depth_map[Yp, Xp]

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{class_name} {confidence} Coord:({Xp},{Yp},{z_value:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            points[class_name] = (Xp, Yp)

    # Dibujar línea entre drumsticks_mid y drumsticks_tip
    if "drumsticks_mid" in points and "drumsticks_tip" in points:
        cv2.line(img, points["drumsticks_mid"], points["drumsticks_tip"], (0, 255, 0), 2)
        (X, Y) = points["drumsticks_tip"]
        fix_position=visualizer.update(raw_data=imu_data)
        print("fix_position: ", fix_position)
        
        midi_note = map_position_to_midi(X, Y)
        send_midi_note(midi_note)

    cv2.imshow("Cam", img)

    if cv2.waitKey(1) == ord('q'):
        sensor_thread.stop()
        cap.release()
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