# Importar librerías
from ultralytics import YOLO  # Importar el modelo YOLO de Ultralytics
import cv2                   # Importar la biblioteca OpenCV
import math                  # Importar el módulo math para operaciones matemáticas

#extremo superior izquierdo es el (0,0)
#extremo superior derecho es el (640,480)
#X aumenta de izquirda a derecha
#Y aumenta de arriba hacia abajo

# Iniciar la webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # Abrir la cámara predeterminada (índice 0)
cap.set(3, 640)               # Establecer el ancho del fotograma en 640 píxeles
cap.set(4, 480)               # Establecer la altura del fotograma en 480 píxeles

# Cargar el modelo YOLO
model = YOLO("yolo-Weights/best_yolo11m_v2.pt")  # Cargar el modelo YOLOv11 entrenado

# Definir las clases de objetos para la detección

classNames = ["drumsticks_mid","drumsticks_tip"]

# Abrir archivo para guardar detecciones
with open("detections.txt", "w") as file:
    file.write("Clase, Coordenada_X, Coordenada_Y, Coordenada_Z, Area\n")  # Encabezado del archivo

# Bucle infinito para capturar continuamente fotogramas de la cámara
while True:
    # Leer un fotograma de la cámara
    success, img = cap.read()

    if not success:
        break

    # Realizar la detección de objetos utilizando el modelo YOLO en el fotograma capturado
    results = model.predict(img, conf=0.6, stream=True)
    points = {}

    # Iterar a través de los resultados de la detección de objetos
    for r in results:
        boxes = r.boxes  # Extraer las cajas delimitadoras de los objetos detectados

        # Iterar a través de cada caja delimitadora
        for box in boxes:

            # Coordenadas de la caja delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            Xp = (x1 + x2) // 2  # Coordenada X central
            Yp = (y1 + y2) // 2  # Coordenada Y central
            area = (x2 - x1) * (y2 - y1)
            

            # Obtener la confianza y clase
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name == "drumsticks_tip":
                Zp = 793/area
            else:
                Zp = 368/area

            # Dibujar la caja y el texto en el fotograma
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{class_name} conf:{confidence} Coord:({Xp},{Yp},{Zp:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Guardar en archivo txt
            with open("detections.txt", "a") as file:
                file.write(f"{class_name}, {Xp}, {Yp}, {Zp}, {area}\n")

            # Guardar puntos para dibujar línea
            points[class_name] = (Xp, Yp)

    # Dibujar línea entre "drumsticks_mid" y "drumsticks_tip" si ambos están detectados
    if "drumsticks_mid" in points and "drumsticks_tip" in points:
        cv2.line(img, points["drumsticks_mid"], points["drumsticks_tip"], (0, 255, 0), 2)

    # Mostrar el fotograma
    cv2.imshow("Cam", img)

    # Salir con 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la cámara
cap.release()

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()