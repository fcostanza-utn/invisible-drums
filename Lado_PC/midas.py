import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimator:
    def __init__(self, device=None):
        self.midas = "Intel/dpt-hybrid-midas"
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DPTForDepthEstimation.from_pretrained(self.midas).to(self.device)
        self.feature_extractor = DPTImageProcessor.from_pretrained(self.midas)

    # Función para estimar profundidad usando MiDaS
    def estimate_depth(self,image):
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Convertir la profundidad a formato numpy y normalizar
        depth = predicted_depth.squeeze().cpu().numpy()
        depth_min, depth_max = depth.min(), depth.max()
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)

        # Redimensionar el mapa de profundidad para que coincida con el tamaño de la imagen de entrada
        depth_resized = cv2.resize(normalized_depth, (image.shape[1], image.shape[0]))

        return depth_resized
    
    def ConvertToAbsoluteDepth(self, depth_map, calibration_poly):
        # Aplicar la función de calibración para obtener la profundidad absoluta
        absolute_depth_map = np.polyval(calibration_poly, depth_map)
        return absolute_depth_map