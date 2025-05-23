import sys
import os
from ultralytics import YOLO
from utils.utilities import get_root_dir
from single_11 import evaluate_single_11
from batch_11 import evaluate_batch_11

if __name__ == "__main__":
    args = sys.argv[1:]
    script_dir = get_root_dir()
    model_path = "weights.pt"
    model_path = os.path.join(script_dir, model_path)
    model = YOLO(model_path)

    image_name = "img_name"
    batch_folder = "batch_name"

    if len(args) == 0:
        sys.exit(1)

    mode = args[0].lower()

    if mode == "single":
        show_results = "verbose" in args
        evaluate_single_11(image_name, show_results=show_results, model=model)
    elif mode == "batch":
        evaluate_batch_11(model, batch_folder, log=True, std=True, set=None)
    else:
        print("Unknown mode. Use 'single' or 'batch'.")
