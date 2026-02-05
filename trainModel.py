from ultralytics import YOLO

# use pretrained yolov8nano model
# make sure datasets folder is in same folder as this
model = YOLO('yolov8n.pt')

model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    workers=16
)
