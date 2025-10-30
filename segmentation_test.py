from ultralytics import YOLO

# Load a pretrained segmentation model
model = YOLO("yolo11n-seg.pt")

results = model("https://ultralytics.com/images/bus.jpg")

results[0].show()
