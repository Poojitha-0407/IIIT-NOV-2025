from ultralytics import YOLO

# Load pretrained YOLO model (small version)
model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/zidane.jpg")

results[0].show()

results = model("https://ultralytics.com/images/zidane.jpg", save=True)
