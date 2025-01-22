from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # build a new model from YAML

results = model.train(
    data='D:/Google Drive/Facultad/Codigo Proyecto/dataset/data.yaml',  # Ruta a tu archivo dataset.yaml
    epochs=20,                            # Número de épocas
    batch=6,                             # Tamaño del lote
    imgsz=640,                            # Tamaño de las imágenes
    project='runs/train',                 # Carpeta para guardar los resultados
    name='drumsticks',            # Nombre del experimento
    device=0                              # Usa GPU (o 'cpu' para usar el procesador)
)

print(results)
