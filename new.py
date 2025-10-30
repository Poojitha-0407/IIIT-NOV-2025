from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

results = model("https://ultralytics.com/images/zidane.jpg")

results[0].show()

results = model("https://ultralytics.com/images/zidane.jpg", save=True)
