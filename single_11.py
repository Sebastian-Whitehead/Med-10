from math import e
from pyexpat import model
import re
import supervision as sv
from supervision.metrics import MeanAveragePrecision
import numpy as np
import os
import cv2
from utils.logger import log_results_to_json
from utils.utilities import annotate_and_display11, get_root_dir, update_batch_id
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

def resize_it(image, size):
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = int(size * aspect_ratio)
    return cv2.resize(image, (new_width, size), interpolation=cv2.INTER_AREA)

def get_image_anno_path_x(image_name):
    root_dir = get_root_dir()
    image_path = os.path.join(root_dir, "batch_images", "test_set", "images", image_name)
    base_name, _ = os.path.splitext(image_name)
    annotation_path = os.path.join(root_dir, "batch_images", "test_set", "labels", f"{base_name}.txt")
    return image_path, annotation_path

def evaluate_single_11(image_name, show_results=False, log=False, sali=False, model_path=None, model=None):
    script_dir = get_root_dir()
    if model_path is not None:
        model_path = os.path.join(script_dir, model_path)
        model = YOLO(model_path)
    elif model is None:
        print("No model path or model passed")
        return None

    map_metric = MeanAveragePrecision()
    image_path, annotation_path = get_image_anno_path_x(image_name)
    image = cv2.imread(image_path)
    image = resize_it(image, 640)
    results = model(image)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
    else:
        print(results[0].boxes)
        print("No valid detections found.")
        return 0, 0

    with open(annotation_path, 'r') as file:
        annotations = []
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = (x_center - width / 2) * image.shape[1]
            y_min = (y_center - height / 2) * image.shape[0]
            x_max = (x_center + width / 2) * image.shape[1]
            y_max = (y_center + height / 2) * image.shape[0]
            annotations.append([x_min, y_min, x_max, y_max, class_id])

    annotations = np.array(annotations)
    xyxy = annotations[:, :4]
    class_id = annotations[:, 4].astype(int)
    confidence = np.ones(len(class_id))

    ground_truth_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id
    )

    map_metric.update(detections, ground_truth_detections)
    eval_metric = map_metric.compute()

    print(f"mAP@50: {eval_metric.map50}")

    if show_results:
        annotate_and_display11(image, detections, ground_truth_detections)
    if log:
        log_results_to_json(
            update_batch_id(), image_name, eval_metric.map50, eval_metric.map50_95,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None
        )
    if eval_metric.map50_95 is None:
        return 0, 0
    return eval_metric.map50_95, eval_metric.map50

if __name__ == "__main__":
    script_dir = get_root_dir()
    model_path = "fine_tune_sets/0/0/weights/best.pt"
    model_path = os.path.join(script_dir, model_path)
    model = YOLO(model_path)
    map_50 = []
    map_95 = []
    for idx, image in enumerate(os.listdir(os.path.join(script_dir, "batch_images", "test_set", "images"))):
        print(f'Image {idx}')
        s, d = evaluate_single_11(image, show_results=True, log=False, sali=True, model=model)
        map_50.append(d)
        map_95.append(s)
        print("--------------------------------------------------")
    avg_50 = np.average(map_50)
    avg_95 = np.average(map_95)
    print(f"Average mAP@50: {avg_50}")
    print(f"Average mAP@50:95: {avg_95}")
