from ultralytics import YOLO

# Load model   
model = YOLO('models/best.pt')

# Test model    
results = model.predict("input_videos/CAR_vs_NYR_001.mp4", save = True)

# Print results 
print(results[0])
print("\n#############################################\n")

for box in results[0].boxes:
    print(box)