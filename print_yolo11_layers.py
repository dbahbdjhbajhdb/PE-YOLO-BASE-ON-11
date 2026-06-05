from ultralytics import YOLO

model = YOLO("yolo11n.pt")
net = model.model

for i, m in enumerate(net.model):
    print(f"{i}: {m.__class__.__name__}")
    print(m)
    print("-" * 80)
