from ultralytics import YOLO

model = YOLO('runs/detect/train2/weights/best.pt')

results = model.val(data='data.yaml', split='test')
print(results)

