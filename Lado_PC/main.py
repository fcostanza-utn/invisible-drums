#librerías
from ultralytics import YOLO
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mido
from imu_module import IMUVisualizer
from midas import DepthEstimator
import pyqtgraph as pg
import sys
import time
"""
INICIALIZACION DE YOLO
"""
# modelo YOLO
model_yolo = YOLO("yolo-Weights/best_yolo11m_v2.pt")
classNames = ["drumsticks_mid", "drumsticks_tip"] # Definir las clases de objetos para la detección
# Midas
depth_estimator = DepthEstimator()
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
    if x < 100 and y < 100:   # Condicion
        return 36  # Nota MIDI para un kick drum
    elif x < 200 and y < 200: # Condicion 
        return 38  # Nota MIDI para un snare drum
    elif x < 200 and y < 200: # Condicion 
        return 42  # Nota MIDI para un hihat drum
    elif x < 200 and y < 200: # Condicion 
        return 49  # Nota MIDI para un crash drum
    elif x < 200 and y < 200: # Condicion 
        return 51  # Nota MIDI para un ride drum
    elif x < 200 and y < 200: # Condicion 
        return 50  # Nota MIDI para un hightom drum
    elif x < 200 and y < 200: # Condicion 
        return 45  # Nota MIDI para un lowtom drum
    return None

"""
BUCLE PRINCIPAL
"""
def update():
    start_time = time.time()
    imu_return = visualizer.update()
    end_time = time.time()
    
    success, img = cap.read()
    if not success:
        return

    #start_time = time.time()
    depth_map = depth_estimator.estimate_depth(img)
    #end_time = time.time()
    results = model_yolo.predict(img, conf=0.6, stream=True)
    points = {}  # Diccionario para almacenar puntos centrales y profundidad por clase

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Coordenadas de la caja delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            Xp, Yp = (x1 + x2) // 2, (y1 + y2) // 2  # Coordenada central de la caja
            z_value = depth_map[Yp, Xp]

            # confianza y clase
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Dibujar la caja y el texto en el fotograma
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{class_name} {confidence} Coord:({Xp},{Yp},{z_value:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if class_name == "drumsticks_tip":
                tip_points.append((Xp, Yp, z_value))
            elif class_name == "drumsticks_mid":
                mid_points.append((Xp, Yp, z_value))

            # Guardar en archivo txt
            #with open("detections.txt", "a") as file:
                #file.write(f"{class_name}, {Xp}, {Yp}, {z_value:.4f}\n")

            # Guardar puntos para dibujar línea
            points[class_name] = (Xp, Yp)

    # Dibujar línea entre "drumsticks_mid" y "drumsticks_tip" si ambos están detectados
    if "drumsticks_mid" in points and "drumsticks_tip" in points:
        cv2.line(img, points["drumsticks_mid"], points["drumsticks_tip"], (0, 255, 0), 2)
        (X,Y) = points["drumsticks_tip"]
        midi_note = map_position_to_midi(X,Y)
        send_midi_note(midi_note)

    # Mostrar el fotograma
    cv2.imshow("Cam", img)
    wait_time = end_time - start_time
    print(f"Tiempo de espera entre update's: {wait_time:.4f} segundos")
    print("imu_return: ", imu_return)

with open("detections.txt", "w") as file:
    file.write("Clase, Coordenada_X, Coordenada_Y, Profundidad\n")  # Encabezado del archivo

#Cunfiguro timer
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(17))
visualizer.view.show()

while True:
    if cv2.waitKey(1) == ord('q'):
        break
"""
Liberacion de recursos
"""
sys.exit(visualizer.app.exec_())
cap.release()
cv2.destroyAllWindows()
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