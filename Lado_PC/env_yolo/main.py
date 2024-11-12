# Importar librerías
from ultralytics import YOLO
import cv2
import math
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mido

# Configurar el dispositivo para CUDA si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo YOLO y MiDaS
model_yolo = YOLO("yolo-Weights/best_yolo11m_v2.pt")
model_midas = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

# Definir las clases de objetos para la detección
classNames = ["drumsticks_mid", "drumsticks_tip"]


# Iniciar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)  # Ancho del fotograma
cap.set(4, 480)  # Alto del fotograma

tip_points = []
mid_points = []

# Función para estimar profundidad usando MiDaS
def estimate_depth(image):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_midas(**inputs)
        predicted_depth = outputs.predicted_depth

    # Convertir la profundidad a formato numpy y normalizar
    depth = predicted_depth.squeeze().cpu().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)

    # Redimensionar el mapa de profundidad para que coincida con el tamaño de la imagen de entrada
    depth_resized = cv2.resize(normalized_depth, (image.shape[1], image.shape[0]))

    return depth_resized

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

# Función para enviar una nota MIDI
def send_midi_note(note):
    if note:
        midi_out.send(mido.Message('note_on', note=note, velocity=100))
        midi_out.send(mido.Message('note_off', note=note, velocity=100, time=0.1))  # La apaga después de un tiempo breve

print("Available MIDI output ports:")
print(mido.get_output_names())

portmidi = mido.Backend('mido.backends.rtmidi')
midi_out = portmidi.open_output('MIDI1 1')

# Abrir archivo para guardar detecciones
with open("detections.txt", "w") as file:
    file.write("Clase, Coordenada_X, Coordenada_Y, Profundidad\n")  # Encabezado del archivo

# Bucle para capturar fotogramas
while True:
    success, img = cap.read()
    if not success:
        break

    # Estimar el mapa de profundidad del fotograma completo
    depth_map = estimate_depth(img)

    # Realizar la detección de objetos usando YOLO
    results = model_yolo.predict(img, conf=0.6, stream=True)
    points = {}  # Diccionario para almacenar puntos centrales y profundidad por clase

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Coordenadas de la caja delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            Xp, Yp = (x1 + x2) // 2, (y1 + y2) // 2  # Coordenada central de la caja

            # Obtener la profundidad desde el mapa de profundidad
            z_value = depth_map[Yp, Xp]

            # Obtener la confianza y clase
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
            with open("detections.txt", "a") as file:
                file.write(f"{class_name}, {Xp}, {Yp}, {z_value:.4f}\n")

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

    # Salir con 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
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