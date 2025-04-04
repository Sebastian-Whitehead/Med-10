from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall
from utils.logger import log_results_to_json
from utils.utilities import get_root_dir
from ultralytics import YOLO
import cv2
import os

batch_id = datetime.now().strftime("%m%d%H%M%S")

def evaluate_batch_11(batch_path, log = False):
    print("")
    print("YOLO v11")
    print("")
    model_path = "weights.pt"

    model = YOLO(model_path)
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
    results = []
    p = 0
    for path in image_paths:
        print(path)
        image = cv2.imread(path)
        result = model(image)
        xyxy = result[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = result[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result[0].boxes.cls.cpu().numpy().astype(int)  # Class indices
        detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
        results.append(detections)

    for idx, result in enumerate(results):
        annotations = annotations_list[idx]
        map_metric.update(result, [annotations])
        precision_metric.update(result, [annotations])
        recall_metric.update(result, [annotations])

    eval_metric = map_metric.compute()
    pre = precision_metric.compute()
    rec = recall_metric.compute()
    print(f"Precision: {pre.precision_at_50}")
    print(f"Recall: {rec.recall_at_50}")
    print(f"Mean Average Precision for batch: {eval_metric.map50}")

    if log:
        log_results_to_json(batch_id, batch_path, eval_metric.map50, eval_metric.map50_95, pre.precision_at_50, rec.recall_at_50)

    return eval_metric.map50

if __name__ == "__main__":
    batch_folder = "batch_images/test_set"

    evaluate_batch_11(batch_folder, log=True)