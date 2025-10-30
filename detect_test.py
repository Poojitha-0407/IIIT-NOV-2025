from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\DELL\Desktop\IIITH\runs\detect\train\weights\best.pt")

results = model("https://ultralytics.com/images/bus.jpg")

results[0].show()
