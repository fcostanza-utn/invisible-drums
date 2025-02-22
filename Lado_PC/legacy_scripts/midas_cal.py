import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

#midas = "Intel/dpt-large"
midas = "Intel/dpt-hybrid-midas"
# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo DPT-Hybrid-MiDaS y el procesador de imágenes correspondiente
model = DPTForDepthEstimation.from_pretrained(midas).to(device)
image_processor = DPTImageProcessor.from_pretrained(midas)

def estimate_depth(image):
    """
    Estima la profundidad relativa a partir de la imagen utilizando DPT-Hybrid-MiDaS.
    La salida se normaliza en el rango [0, 1].
    """
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    depth = predicted_depth.squeeze().cpu().numpy()  # Valores relativos
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    return normalized_depth

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Ancho
cap.set(4, 480)  # Alto

# Variables de calibración
calibration_points = []  # Lista de tuplas (relative_depth_value, actual_distance)
calibrated = False
calibration_poly = None  # Coeficientes del polinomio de calibración

print("Presiona 'c' para capturar un punto de calibración (tomando la profundidad del centro del frame).")
print("Presiona 'f' para finalizar la calibración (se requiere al menos 2 puntos).")
print("Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener mapa de profundidad relativo
    depth_map = estimate_depth(frame)
    # Redimensionar (aunque normalmente depth_map ya coincide con el tamaño del frame)
    depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Si ya se calibró, aplicar la función de calibración para obtener profundidad absoluta
    if calibrated and calibration_poly is not None:
        # np.polyval aplica el polinomio a cada elemento del mapa
        absolute_depth_map = np.polyval(calibration_poly, depth_map_resized)
        display_map = absolute_depth_map
    else:
        display_map = depth_map_resized

    # Mostrar el mapa de profundidad (escala original o calibrada)
    cv2.imshow("Depth Map", display_map)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Tomar el valor de profundidad en el centro de la imagen
        h, w = depth_map_resized.shape
        center_val = depth_map_resized[h // 2, w // 2]
        print(f"Valor de profundidad relativo en el centro: {center_val:.4f}")
        try:
            # Pedir al usuario la distancia real en metros para ese punto
            actual_distance = float(input("Ingrese la distancia real (en metros) para el centro del frame: "))
            calibration_points.append((center_val, actual_distance))
            print(f"Punto calibrado: profundidad relativo={center_val:.4f}, distancia real={actual_distance}")
        except Exception as e:
            print("Error al leer la distancia real:", e)
    elif key == ord('f'):
        if len(calibration_points) >= 2:
            # Separar los datos de calibración
            relative_vals = np.array([pt[0] for pt in calibration_points])
            actual_distances = np.array([pt[1] for pt in calibration_points])
            # Ajustar un polinomio de grado 2 (puedes probar grado 1 o mayor según sea necesario)
            calibration_poly = np.polyfit(relative_vals, actual_distances, 2)
            calibrated = True
            print("Calibración finalizada. Coeficientes del polinomio:", calibration_poly)
        else:
            print("Se requieren al menos 2 puntos de calibración para finalizar.")

cap.release()
cv2.destroyAllWindows()
