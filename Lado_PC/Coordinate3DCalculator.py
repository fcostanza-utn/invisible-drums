import numpy as np

class Coordinate3DCalculator:
    def __init__(self, focal_length_mm=7.8, sensor_width_mm=9.40741, image_width_pixels=640, 
                 known_distance_cm=20):
        """
        Inicializa el calculador de coordenadas 3D
        
        Args:
            focal_length_mm: Distancia focal de la cámara en milímetros
            sensor_width_mm: Ancho del sensor de la cámara en milímetros
            image_width_pixels: Ancho de la imagen en píxeles
            known_distance_cm: Distancia conocida entre la punta azul y la cinta roja en centímetros
        """
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        self.image_width_pixels = image_width_pixels
        self.known_distance_cm = known_distance_cm
        
        # Calcular factor de conversión de píxeles a mm
        self.pixels_to_mm = sensor_width_mm / image_width_pixels
        
    def calculate_coordinates(self, blue_tip_px, red_tape_px, image_height_pixels):
        """
        Calcula las coordenadas 3D en centímetros
        
        Args:
            blue_tip_px: Tupla (x, y) de la posición de la punta azul en píxeles
            red_tape_px: Tupla (x, y) de la posición de la cinta roja en píxeles
            image_height_pixels: Altura de la imagen en píxeles
            
        Returns:
            Tupla (x, y, z) con las coordenadas en centímetros
        """
        # Convertir coordenadas a numpy arrays
        blue_tip = np.array(blue_tip_px)
        red_tape = np.array(red_tape_px)
        
        # Calcular distancia en píxeles entre los dos puntos
        pixel_distance = np.linalg.norm(blue_tip - red_tape)
        
        # Calcular profundidad (Z) usando similitud de triángulos
        # known_distance_cm * focal_length = pixel_distance * Z
        z_cm = (self.known_distance_cm * self.focal_length_mm) / (pixel_distance * self.pixels_to_mm)
        
        # Calcular coordenadas X e Y
        # Convertir coordenadas de imagen a coordenadas centradas
        x_centered = blue_tip[0] - self.image_width_pixels / 2
        y_centered = blue_tip[1] - image_height_pixels / 2
        
        # Convertir a centímetros usando la profundidad calculada
        x_cm = (x_centered * self.pixels_to_mm * z_cm) / (self.focal_length_mm)
        y_cm = (y_centered * self.pixels_to_mm * z_cm) / (self.focal_length_mm)
        
        return x_cm, y_cm, z_cm
    
    def get_confidence_metrics(self, blue_tip_px, red_tape_px):
        """
        Calcula métricas de confianza para la estimación
        
        Returns:
            Dict con métricas de confianza
        """
        pixel_distance = np.linalg.norm(np.array(blue_tip_px) - np.array(red_tape_px))
        
        metrics = {
            'pixel_distance': pixel_distance,
            'relative_depth_uncertainty': abs(1 - pixel_distance / (self.image_width_pixels / 4)),
        }
        
        return metrics