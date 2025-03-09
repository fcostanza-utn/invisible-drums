import cv2
import numpy as np
from openni import openni2

openni2.initialize("C:/Program Files/OpenNI2/Redist")
dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()

depth_stream.start()
color_stream.start()

while True:
    # Capturar frames
    depth_frame = depth_stream.read_frame()
    color_frame = color_stream.read_frame()

    # Convertir a arrays de NumPy
    depth_data = np.array(depth_frame.get_buffer_as_uint16()).reshape((depth_frame.height, depth_frame.width))
    color_data = np.array(color_frame.get_buffer_as_triplet()).reshape((color_frame.height, color_frame.width, 3))

    # Convertir RGB a BGR (porque OpenCV usa BGR por defecto)
    color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

    # Mostrar imágenes
    cv2.imshow("Depth", depth_data)
    cv2.imshow("Color", color_data)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
        break
