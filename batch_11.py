import sys
import os
from datetime import datetime
from networkx import bidirectional_dijkstra
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall
from utils.logger import log_results_to_json
from utils.utilities import get_root_dir
from utils.utilities import update_batch_id
from ultralytics import YOLO
import cv2
import time
from tqdm import tqdm
import numpy as np

def resize_it(image, size):

    original_height, original_width = image.shape[:2]

    aspect_ratio = original_width / original_height
    new_width = int(size * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, size), interpolation=cv2.INTER_AREA)
    return resized_image

def evaluate_batch_11(model, batch_path, log = False, std = False, set=None):
    print("")
    print("YOLO v11")
    print("")

    import logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


    precision_metric = Precision()
    recall_metric = Recall()
    map_metric = MeanAveragePrecision()

    root_dir = get_root_dir()
    batch_path = os.path.join(root_dir, batch_path)
    data_set = sv.DetectionDataset.from_yolo(
        images_directory_path=f'{batch_path}/images',
        annotations_directory_path=f'{batch_path}/labels',
        data_yaml_path=f'{batch_path}/data.yaml'
    )

    image_paths = [data_set[idx][0] for idx in range(len(data_set))]
    annotations_list = [data_set[idx][2] for idx in range(len(data_set))]

    print(f"Evaluating batch of {len(image_paths)} images")
    results = []
    start_time = time.time()
    for path in tqdm(image_paths, desc="Processing Images"):
        image = cv2.imread(path)
        image = resize_it(image, 640)
        result = model(image)
        xyxy = result[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = result[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result[0].boxes.cls.cpu().numpy().astype(int)  # Class indices
        class_ids = np.zeros(len(class_ids), dtype=int)
        detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
        results.append(detections)

    end_time = time.time()
    time_total = end_time - start_time
    time_per_image = (end_time - start_time) / len(image_paths)
    if std:
        individual_map_values_50 = []  # List to store mAP@50 values for each image
        individual_map_values_95 = []  # List to store mAP@50:95 values for each image
        recall_values = []
        precision_values = []
        for idx, result in enumerate(results):
            annotations = annotations_list[idx]
            annotations.class_id = np.zeros(len(annotations.class_id), dtype=int)
            map_metric.update(result, [annotations])
            precision_metric.update(result, [annotations])
            recall_metric.update(result, [annotations])
            eval_metric = map_metric.compute()
            individual_map_values_50.append(eval_metric.map50) 
            individual_map_values_95.append(eval_metric.map50_95) 
            recall_values.append(recall_metric.compute().recall_at_50)
            precision_values.append(precision_metric.compute().precision_at_50)
            map_metric.reset()
            recall_metric.reset()
            precision_metric.reset()
        std_dev_50 = np.std(individual_map_values_50)
        std_dev_95 = np.std(individual_map_values_95)
        std_recall = np.std(recall_values)
        std_precision = np.std(precision_values)
    t = 0
    for idx, result in enumerate(results):
        if len(result.class_id) == 0:
            t += 1
            continue
        annotations = annotations_list[idx]
        map_metric.update(result, [annotations])
        precision_metric.update(result, [annotations])
        recall_metric.update(result, [annotations])
    print(f"Number of images with no valid detections: {t}")
    eval_metric = map_metric.compute()
    pre = precision_metric.compute()
    rec = recall_metric.compute()

    if log:
        log_results_to_json(update_batch_id(), batch_path, "11", set, eval_metric.map50, eval_metric.map50_95, pre.precision_at_50, rec.recall_at_50, eval_metric.matched_classes, eval_metric.ap_per_class, std_dev_50, std_dev_95, std_recall, std_precision, time_total, time_per_image)

    return eval_metric.map50

if __name__ == "__main__":
    model_path = "weights.pt"
    model_path = os.path.join(get_root_dir(), model_path)
    model = YOLO(model_path)

    batch_folder = "data"
    for set in os.listdir(batch_folder):
        set_folder = os.path.join(get_root_dir(), batch_folder, set)
        print(set)
        print(set_folder)
        if set == "2":
            evaluate_batch_11(model, set_folder, log=True, std=True, set=set)
        else:
            continue
            evaluate_batch_11(model, set_folder, log=True, std=True, set=set)
