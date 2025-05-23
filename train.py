import os
import sys
from ultralytics import YOLO
from utils.utilities import get_root_dir

def train(set_name):
    model = YOLO("weights.pt")
    data_path = os.path.join(get_root_dir(), "fine_tune_sets", set_name, "train", "data.yaml")
    model.train(
        data=data_path,
        epochs=20,
        imgsz=640,
        device="mps",
        pretrained=True,
        save=True,
        degrees=180,
        flipud=0.5,
        fliplr=0.5,
        project=os.path.join(get_root_dir(), "fine_tune_sets", set_name),
        name=set_name,
        exist_ok=True
    )

if __name__ == "__main__":
    set_name = sys.argv[1] if len(sys.argv) > 1 else "0"
    train(set_name)