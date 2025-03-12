import json
import os
from datetime import datetime

def log_results_to_json(batch_id, batch_path, map50, map50_95, precision, recall):
    log_data = {
        "batch_id": batch_id,
        "batch_path": batch_path,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall
    }

    # Create the results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
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