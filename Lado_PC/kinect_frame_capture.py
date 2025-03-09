import cv2
import numpy as np
from openni import openni2
import os
import time

def main():
    # Inicializa OpenNI2 (ajusta la ruta según tu sistema; en Windows suele ser "C:/Program Files/OpenNI2/Redist")
    # En Linux, asegúrate de que OpenNI2 esté instalado y la ruta sea la correcta.
    openni2.initialize("C:/Program Files/OpenNI2/Redist")
    mode = openni2.VideoMode(
        pixelFormat=openni2.PIXEL_FORMAT_RGB888,  # Asegura que se use un formato de 24 bits
        resolutionX=640,
        resolutionY=480,
        fps=60
    )
    dev = openni2.Device.open_any()

    # Crear y arrancar el stream de color
    color_stream = dev.create_color_stream()
    color_stream.set_video_mode(mode)
    color_stream.start()

    # (Opcional) Configura el modo de salida de la imagen
    # Por ejemplo, para asegurarte de obtener una imagen en RGB:
    # color_stream.set_video_mode(openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))

    # Crear la carpeta para guardar las imágenes si no existe
    dataset_dir = "C:/Users/fgcos/OneDrive/Pictures/fotos_kinect"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    frame_count = 3250
    print("Capturando frames. Presiona 'q' en la ventana de imagen para salir.")

    while True:
        # Capturar frame de color
        color_frame = color_stream.read_frame()
        color_data = np.array(color_frame.get_buffer_as_triplet()).reshape(
            (color_frame.height, color_frame.width, 3))
        # Convertir de RGB a BGR (OpenCV utiliza BGR por defecto)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
        
        # Guardar la imagen en la carpeta
        filename = os.path.join(dataset_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, color_data)
        frame_count += 1
        
        # Mostrar la imagen (opcional)
        cv2.imshow("Color Frame", color_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Puedes agregar un pequeño sleep si necesitas reducir la velocidad de captura
        # time.sleep(0.033)  # ~30 fps

    # Detener el stream y liberar recursos
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
