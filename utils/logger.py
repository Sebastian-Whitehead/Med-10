import json
import os
from utils.utilities import get_root_dir

def format_list_items(data):
    if isinstance(data, list):
        return [format_list_items(item) for item in data]
    elif isinstance(data, float):
        return round(data, 4) 
    else:
        return data 
    
def log_results_to_json(batch_id, batch_path, model, test, map50, map50_95, precision, recall, matched_classes, AP_per_class, std_dev_50, std_dev_95, std_recall, std_precision, time_total, time_per_image, image_count):
    log_data = {
        "batch_id": batch_id,
        "batch_path": batch_path,
        "model": model,
        "test": test,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "matched_classes": matched_classes.tolist(),
        "AP_per_class": AP_per_class.tolist(),
        "std_recall": std_recall,
        "std_precision": std_precision,
        "std_dev_50": std_dev_50,
        "std_dev_95": std_dev_95,

        "time_total": time_total,
        "time_per_image": time_per_image,
        "image_count": image_count
    }

    results_dir = os.path.join(get_root_dir(), "fine_tune_sets", "results")
    os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join(results_dir, f"ev_{batch_id}.json")

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            existing_data = json.load(f)
        if isinstance(existing_data, list):
            existing_data.append(log_data)
        else:
            existing_data = [existing_data, log_data]
    else:
        existing_data = [log_data]

    with open(log_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Results logged to {log_file}")