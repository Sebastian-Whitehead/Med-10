from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall
#from inference import get_model
import numpy as np
import os
from PIL import Image
from logger import log_results_to_json
from ultralytics import YOLO
import cv2
from map import log_results_to_json

batch_id = datetime.now().strftime("%m%d%H%M%S")

def evaluate_batch_11(batch_path):
    print("")
    print("YOLO v11")
    print("")
    model_path = "weights.pt"  # Replace with the actual path

    model = YOLO(model_path)
    precision_metric = Precision()
    recall_metric = Recall()
    map_metric = MeanAveragePrecision()

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
        print(result)
        annotations = annotations_list[idx]
        #xyxy_annotations, class_ids = process_annotations(annotations)
        #annotation_detections = sv.Detections(xyxy=xyxy_annotations, class_id=class_ids)
        
        

        

        map_metric.update(result, [annotations])
        precision_metric.update(result, [annotations])
        recall_metric.update(result, [annotations])

    #print("Predictions stored in map_metric:")
    #print(map_metric._predictions_list)

    #print("\nTargets (Annotations) stored in map_metric:")
    #print(map_metric._targets_list)


    eval_metric = map_metric.compute()
    pre = precision_metric.compute()
    rec = recall_metric.compute()
    print(f"Precision: {pre.precision_at_50}")
    print(f"Recall: {rec.recall_at_50}")
    print(f"Mean Average Precision for batch: {eval_metric.map50}")

    # Log results to JSON file
    log_results_to_json(batch_id, batch_path, eval_metric.map50, eval_metric.map50_95, pre.precision_at_50, rec.recall_at_50)

    return eval_metric.map50


def xyxy_to_cxcywh(xyxy):
    xyxy = np.array(xyxy)
    x_min, y_min, x_max, y_max = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return np.stack((cx, cy, w, h), axis=1)

def cxcywh_to_xyxy(cxcywh):
    cxcywh = np.array(cxcywh)
    cx, cy, w, h = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return np.stack((x_min, y_min, x_max, y_max), axis=1)

def process_annotations(annotations):
    converted_annotations = []
    class_ids = []
    
    for anno in annotations:
        cxcywh, _, _, class_id, _, _ = anno  # Extract relevant values
        xyxy = cxcywh_to_xyxy(np.array([cxcywh]))[0]  # Convert format
        converted_annotations.append(xyxy)
        class_ids.append(class_id)
    
    return np.array(converted_annotations, dtype=np.float32), np.array(class_ids, dtype=np.int32)
