import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

# Cargar el modelo MiDaS de Hugging Face y configurarlo para usar la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")

def estimate_depth(image):
    # Preprocesar la imagen para el modelo
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Convertir la profundidad a formato numpy y normalizar
    depth = predicted_depth.squeeze().cpu().numpy()  # Traer a la CPU para convertir a numpy
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)

    return normalized_depth

# Usar la función de estimación de profundidad con cada cuadro
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener el mapa de profundidad
    depth_map = estimate_depth(frame)

    #print (depth_map)
    # Redimensionar y mostrar el mapa de profundidad
    depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
    cv2.imshow("Depth Map", depth_map_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
