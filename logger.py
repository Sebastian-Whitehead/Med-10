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

    log_file = os.path.join(batch_path, f"evaluation_results_{batch_id}.json")
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

    print(f"Results logged to {log_file}")