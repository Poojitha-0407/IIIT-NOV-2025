from ultralytics import YOLO
import os

# Load YOLO pretrained models
detect_model = YOLO("yolo11n.pt")        # Object detection
segment_model = YOLO("yolo11n-seg.pt")   # Segmentation

# Folder paths
input_folder = "frames"
detect_output = "results/detect"
segment_output = "results/segment"

# Create output folders if not exist
os.makedirs(detect_output, exist_ok=True)
os.makedirs(segment_output, exist_ok=True)

# Loop through all frames
for img in sorted(os.listdir(input_folder)):
    if img.endswith(".jpg"):
        path = os.path.join(input_folder, img)
        print(f"Processing: {img}")

        # Run detection
        detect_model(path, save=True, project=detect_output, name="", exist_ok=True)

        # Run segmentation
        segment_model(path, save=True, project=segment_output, name="", exist_ok=True)

print("\n✅ All frames processed successfully!")
print("📁 Check 'results/detect' and 'results/segment' folders for output images.")
