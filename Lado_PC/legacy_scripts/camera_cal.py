import cv2
from ultralytics import YOLO
import torch
import math
import numpy as np
import glob
from midas import DepthEstimator

class camera_cal:

    def __init__(self, model_yolo_path="yolo-Weights/best_yolo11m_v3.pt", f_x=0,distancia_referencia=1.0,tamano_real_m=0.18, High=480,Width=640, cal_images=glob.glob('./Fotos_cal/*.jpg'), pattern_size = (9 , 6)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_yolo = YOLO(model_yolo_path).to(self.device)
        self.midas = DepthEstimator(device=self.device)
        self.classNames = ["drumsticks_mid", "drumsticks_tip"]
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, 640)  # Ancho del fotograma
        self.cap.set(4, 480)  # Alto del fotograma
        self.distancia_referencia = distancia_referencia
        self.tamano_real_m = tamano_real_m
        self.f_x = f_x
        self.tamano_cuadrado_px = int((f_x * tamano_real_m) / distancia_referencia)
        self.High = High
        self.Width = Width
        self.fact_esc_prof = []
        self.z_list_cal = []
        self.z_midas_list_cal = []
        self.dist_coef = []                                                                 # Coeficientes de distorsión
        self.intrinsic_matrix = []                                                          # Matriz intrínseca de la cámara
        self.reproy_error = []                                                              # Error de reproyección (RMS)
        self.rot_vector = []                                                                # Vectores de rotación (uno por imagen)
        self.tras_vector = []                                                               # Vectores de traslación (uno por imagen)
        self.objpoints = []                                                                 # Puntos 3D en el sistema del tablero
        self.imgpoints = []                                                                 # Puntos 2D detectados en la imagen
        self.cal_images = cal_images                                                        # Ruta donde estan guardadas las imagenes para calibrar
        self.pattern_size = pattern_size                                                    # Tamaño del patrón (cantidad de esquinas interiores de cada lado)

        self.calibrate_with_pattern()

        print("Tamaño del cuadrado en píxeles:", self.tamano_cuadrado_px)

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

                # # Dibujar y mostrar las esquinas detectadas
                # cv2.drawChessboardCorners(img, self.pattern_size, corners_subpix, ret)
                # cv2.imshow('Esquinas', img)
                # cv2.waitKey(500)

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

    def Calibrate_depth (self):
        cal_step = 0
        center_tip = []
        center_mid = []
        points = {}
        match = False
        x0 = (self.Width) // 2
        y0 = (self.High) // 2
        while True:
            
            #maquina estados
            match = False
            if cal_step == 0: #primer paso centro de la pantalla baqueta vertical
                center_tip = [x0, y0 - 62]
                center_mid = [x0, y0 + 62]
            elif cal_step == 1:
                center_tip = [x0-100, y0 - 62]
                center_mid = [x0-100, y0 + 62]
            elif cal_step == 2:
                center_tip = [x0+100, y0 - 62]
                center_mid = [x0+100, y0 + 62]
            elif cal_step == 3:
                center_tip = [x0, y0 - 40]
                center_mid = [x0, y0 + 40]
            elif cal_step == 4:
                center_tip = [x0, y0 - 80]
                center_mid = [x0, y0 + 80]
            elif cal_step == 5:
                center_tip = [x0 + 100, y0 - 80]
                center_mid = [x0 + 100, y0 + 80]
            elif cal_step == 6:
                center_tip = [x0 - 100, y0 - 80]
                center_mid = [x0 - 100, y0 + 80]
            elif cal_step == 7:
                break
                
                
            print("center_tip:", center_tip)
            print("center_mid:", center_mid)
            while match == False:
                ret, frame = self.cap.read()
                #print("frame shape: ", frame.shape)
                orig = frame.copy()
                if ret:
                    ret = False
                    results = self.model_yolo.predict(orig,conf=0.35, stream=True)
                    depth_map = self.midas.estimate_depth(cv2.cvtColor(orig,cv2.COLOR_BGR2RGB))
                    cv2.circle(frame, (center_tip[0], center_tip[1]), 10, (255, 255, 255), 1)
                    cv2.putText(frame, f"center_tip coord=({center_tip[0]},{center_tip[1]})",
                                (center_tip[0], center_tip[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
                    cv2.circle(frame, (center_mid[0], center_mid[1]), 10, (255, 255, 255), 1)
                    cv2.putText(frame, f"center_mid coord=({center_mid[0]},{center_mid[1]})",
                                (center_mid[0], center_mid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            Xp, Yp = (x1 + x2) // 2, (y1 + y2) // 2
                            confidence = math.ceil((box.conf[0] * 100)) / 100
                            cls = int(box.cls[0])
                            class_name = self.classNames[cls]
                            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            cv2.circle(frame, (Xp, Yp), 5, (0, 255, 0), -1)
                            cv2.putText(frame, f"{class_name} {confidence} Coord:({Xp},{Yp})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
                            points[class_name] = (Xp, Yp)
                    if "drumsticks_mid" in points and "drumsticks_tip" in points:
                        print(points)
                        cv2.line(frame, points["drumsticks_mid"], points["drumsticks_tip"], (0, 255, 0), 1)
                        if (points["drumsticks_mid"][0] > (center_mid[0] - 10)) and (points["drumsticks_mid"][0] < (center_mid[0] + 10)) and points["drumsticks_mid"][1] > center_mid[1] - 10 and points["drumsticks_mid"][1] < center_mid[1] + 10 and points["drumsticks_tip"][0] > center_tip[0] - 10 and points["drumsticks_tip"][0] < center_tip[0] + 10 and points["drumsticks_tip"][1] > center_tip[1] - 10 and points["drumsticks_tip"][1] < center_tip[1] + 10:
                            matrix_aux = (self.intrinsic_matrix[0])
                            matrix_aux_2 = (self.intrinsic_matrix[1])
                            z = self.tamano_real_m *  630 / (points["drumsticks_mid"][1] - points["drumsticks_tip"][1])
                            z_midas = (depth_map[points["drumsticks_tip"][0], points["drumsticks_tip"][1]] + depth_map[points["drumsticks_mid"][0], points["drumsticks_mid"][1]])/2
                            self.fact_esc_prof.append(z/z_midas)
                            self.z_list_cal.append(z)
                            self.z_midas_list_cal.append(z_midas)
                            print("z:", self.z_list_cal)
                            print("depth_map:", depth_map)
                            print("z_midas: ", self.z_midas_list_cal)
                            print("self.fact_esc_prof: ", self.fact_esc_prof)
                            match = True
                            print("match")
                            cal_step += 1
                        else:
                            print("no match")
                            match = False
                            #cal_step = 0        
                    # Mostrar el fotograma
                    cv2.imshow("Frame", frame)
                    frame = orig.copy()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cal = camera_cal()
    cal.Calibrate_depth()