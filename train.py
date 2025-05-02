import os
import sys
from ultralytics import YOLO
from utils.utilities import get_root_dir


def train(set):
    model = YOLO("weights.pt")  
    data_path = os.path.join(get_root_dir(), "fine_tune_sets", set, "train", "data.yaml")
    results = model.train(data=data_path, epochs=20, imgsz=640, device="mps", pretrained=True, save=True, degrees=180, flipud=0.5, fliplr=0.5, 
                          project=os.path.join(get_root_dir(), "fine_tune_sets", set), name=set, exist_ok=True)
    #print(results)

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) > 1:
        set = sys.argv[1]
    else:
        set = 0  # Default set name if not provided
    train(set)