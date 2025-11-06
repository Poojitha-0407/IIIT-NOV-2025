from ultralytics import YOLO

# Load pretrained models
detect_model = YOLO("yolo11n.pt")       # object detection
segment_model = YOLO("yolo11n-seg.pt")  # segmentation

# List of images to process (URLs for now; you can add local paths too)
images = [
    "C:/Users/DELL/Desktop/Pics/Image1.png",
    "C:/Users/DELL/Desktop/Pics/Image2.png",
    "C:/Users/DELL/Desktop/Pics/Image3.png",
    "C:/Users/DELL/Desktop/Pics/Image4.png",
  
]

for img in images:
    print(f"\nProcessing: {img}")

    # Object Detection
    detect_model(img, save=True, project="results/detect")

    # Segmentation
    segment_model(img, save=True, project="results/segment")

print("\n All images processed. Check the 'results' folder for outputs.")

