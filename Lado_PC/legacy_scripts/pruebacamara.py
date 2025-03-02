import cv2
import numpy as np
import glob


class Coordinate3DCalculator:

    def __init__(self, sensor_width_mm=5.37, image_width_pixels=640, cal_images=glob.glob('./Fotos_cal/*.jpg'), pattern_size = (9 , 6)):
        
        self.dist_coef = []                                                                 # Coeficientes de distorsión
        self.intrinsic_matrix = []                                                          # Matriz intrínseca de la cámara
        self.reproy_error = []                                                              # Error de reproyección (RMS)
        self.rot_vector = []                                                                # Vectores de rotación (uno por imagen)
        self.tras_vector = []                                                               # Vectores de traslación (uno por imagen)
        self.objpoints = []                                                                 # Puntos 3D en el sistema del tablero
        self.imgpoints = []                                                                 # Puntos 2D detectados en la imagen
        self.cal_images = cal_images                                                        # Ruta donde estan guardadas las imagenes para calibrar
        self.pattern_size = pattern_size                                                    # Tamaño del patrón (cantidad de esquinas interiores de cada lado)
        self.sensor_width_mm = sensor_width_mm
        self.image_width_pixels = image_width_pixels
        self.pixels_to_mm = sensor_width_mm / image_width_pixels                            # Calcular factor de conversión de píxeles a mm
        
    def calibrate_with_pattern(self):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)          # Parámetros de terminación para refinar la búsqueda de esquinas

        square_size = 2.4                                                                   # Tamaño físico de cada cuadrado

        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)       # Preparar el array de puntos 3D "objp" (en el plano Z=0). 
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size                                                                 # Escalado si se conoce la dimensión real de los cuadrados                        

        for fname in self.cal_images:
            # Leer la imagen
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 7. Buscar las esquinas del tablero de ajedrez
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            # Si se detectaron las esquinas, refinar su ubicación y almacenar
            if ret:
                corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                self.objpoints.append(objp)
                self.imgpoints.append(corners_subpix)

                # Dibujar y mostrar las esquinas detectadas
                cv2.drawChessboardCorners(img, self.pattern_size, corners_subpix, ret)
                cv2.imshow('Esquinas', img)
                cv2.waitKey(500)

        # cv2.destroyAllWindows()

        # Calibrar la cámara usando los puntos recolectados
        self.reproy_error, self.intrinsic_matrix, self.dist_coef, self.rot_vector, self.tras_vector = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        # Imprimir resultados
        # print("Error de reproyección RMS:", self.reproy_error)
        print("Matriz intrínseca (cameraMatrix):\n", self.intrinsic_matrix)
        # print("Coeficientes de distorsión (distCoeffs):\n", self.dist_coef)
        # print("Vectores de rotación (rvecs):", self.rot_vector)
        # print("Vectores de traslación (tvecs):", self.tras_vector)

        # # Guardar los parámetros en un archivo para usarlos después
        # import pickle
        # with open('calibration_data.pkl', 'wb') as f:
        #     pickle.dump((mtx, dist, rvecs, tvecs), f)
    
    def calculate_coordinates(self, blue_tip_px, red_tape_px):

        # Convertir coordenadas a numpy arrays
        blue_tip = np.array(blue_tip_px)
        red_tape = np.array(red_tape_px)
        
        # Calcular distancia en píxeles entre los dos puntos
        pixel_distance = np.linalg.norm(blue_tip - red_tape)
        
        # Calcular profundidad (Z) usando similitud de triángulos
        # known_distance_cm * focal_length = pixel_distance * Z
        focal_prom_px = (self.intrinsic_matrix[1,1] + self.intrinsic_matrix[2,2])/2
        z_cm = (self.known_distance_cm * focal_prom_px) / (pixel_distance)
        
        # Convertir a centímetros usando l  a profundidad calculada
        x_cm = (blue_tip_px[1] - self.intrinsic_matrix[1,3]) * (z_cm / self.intrinsic_matrix[1,1])
        y_cm = (blue_tip_px[2] - self.intrinsic_matrix[2,3]) * (z_cm / self.intrinsic_matrix[2,2])
        
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

# Ejemplo de uso
def prueba():
    # Configuración de la cámara (ajustar según tu cámara)
    calculator = Coordinate3DCalculator(
        sensor_width_mm=5.37,      # Ancho del sensor de tu cámara
        image_width_pixels=640,   # Resolución horizontal de tu cámara
        cal_images = glob.glob('./Fotos_cal/*.jpg'),
        pattern_size = (9 , 6)
    )

    calculator.calibrate_with_pattern()
    
    # Ejemplo de detecciones de YOLO (reemplazar con tus detecciones reales)
    # Las coordenadas están en píxeles (x, y)
    blue_tip_detection = (960, 540)    # Centro de la imagen
    red_tape_detection = (960, 640)    # 100 píxeles más abajo
    image_height = 1080                # Altura total de la imagen
    
    # # Calcular coordenadas 3D
    # x, y, z = calculator.calculate_coordinates(
    #     blue_tip_detection, 
    #     red_tape_detection,
    # )
    
    # Obtener métricas de confianza
    confidence = calculator.get_confidence_metrics(
        blue_tip_detection,
        red_tape_detection
    )
    
    # print(f"Coordenadas de la punta:")
    # print(f"X: {x:.2f} cm")
    # print(f"Y: {y:.2f} cm")
    # print(f"Z: {z:.2f} cm")
    print(f"\nMétricas de confianza:")
    print(f"Distancia en píxeles: {confidence['pixel_distance']:.2f}")
    print(f"Incertidumbre relativa: {confidence['relative_depth_uncertainty']:.2f}")
    print(f"Matriz Intrinseca:")
    print("K: ",calculator.intrinsic_matrix)

if __name__ == "__main__":
    prueba()