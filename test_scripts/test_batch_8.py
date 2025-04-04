from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall
from inference import get_model
import os
from utils.logger import log_results_to_json
from utils.utilities import get_root_dir
from utils.utilities import update_batch_id

def evaluate_batch_8(batch_path, log = False):
    print("")
    print("YOLO v8")
    print("")
    model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")
    precision_metric = Precision()
    recall_metric = Recall()
    map_metric = MeanAveragePrecision()

    root_dir = get_root_dir()
    batch_folder = os.path.join(root_dir, batch_folder)

    data_set = sv.DetectionDataset.from_yolo(
        images_directory_path=f'{batch_path}/images',
        annotations_directory_path=f'{batch_path}/labels',
        data_yaml_path=f'{batch_path}/data.yaml'
    )

    image_paths = [data_set[idx][0] for idx in range(len(data_set))]
    annotations_list = [data_set[idx][2] for idx in range(len(data_set))]

    print(f"Evaluating batch of {len(image_paths)} images")
    results = model.infer(image_paths)

    for idx, result in enumerate(results):
        detections = sv.Detections.from_inference(result)
        annotations = annotations_list[idx]
        map_metric.update(detections, [annotations])
        precision_metric.update(detections, [annotations])
        recall_metric.update(detections, [annotations])

    eval_metric = map_metric.compute()
    pre = precision_metric.compute()
    rec = recall_metric.compute()
    print(f"Precision: {pre.precision_at_50}")
    print(f"Recall: {rec.recall_at_50}")
    print(f"Mean Average Precision for batch: {eval_metric.map50}")

    if log:
        log_results_to_json(update_batch_id, batch_path, eval_metric.map50, eval_metric.map50_95, pre.precision_at_50, rec.recall_at_50)
    return eval_metric.map50

if __name__ == "__main__":
    batch_folder = "batch_images/test_set"
    
    evaluate_batch_8(batch_folder, log=False)