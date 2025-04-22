from math import e
from pyexpat import model
import re
import supervision as sv
from supervision.metrics import MeanAveragePrecision
import numpy as np
import os
import cv2
from utils.logger import log_results_to_json
from utils.utilities import annotate_and_display11
from utils.utilities import get_image_anno_path
from utils.utilities import update_batch_id
from utils.utilities import get_root_dir
from ultralytics import YOLO

import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

def resize_it(image, size):

    original_height, original_width = image.shape[:2]

    aspect_ratio = original_width / original_height
    new_width = int(size * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, size), interpolation=cv2.INTER_AREA)
    return resized_image


def get_image_anno_path_x(image_name):
    root_dir = get_root_dir()
    image_path = os.path.join(root_dir, "data", "3", "images", image_name)
    base_name, _ = os.path.splitext(image_name)
    annotation_path = os.path.join(root_dir, "data", "3", "labels", f"{base_name}.txt")
    return image_path, annotation_path

def evaluate_single_11(image_name, show_results=False, log=False, sali=False, model_path=None, model = None):
    script_dir = get_root_dir()
    if model_path is not None:
        model_path = os.path.join(script_dir, model_path)
        model = YOLO(model_path)
    elif model is not None:
        model = model
    else:
        # error about no model path or model passed
        print("No model path or model passed")
        return None


    map_metric = MeanAveragePrecision()
    image_path, annotation_path = get_image_anno_path_x(image_name)

    image = cv2.imread(image_path)
    image = resize_it(image, 640)  # Resize the image to 640x640
    results = model(image)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        classes = np.zeros(len(class_ids), dtype=int)

        print(f"Class IDs: {class_ids}")
        print(f"Classes: {classes}")
        detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
        detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=classes)
    else:
        print(results[0].boxes)
        print("No valid detections found.")
        return 0, 0

    # Parse YOLO annotation file
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
    classes = np.zeros(len(class_id), dtype=int)
    print(f"Class IDs: {class_id}")
    print(f"Classes: {classes}")
    confidence = np.ones(len(class_id))

    ground_truth_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        #class_id=class_id
        class_id=classes
    )

    print(f"Detections: {detections}")
    print(f"Ground Truth: {ground_truth_detections}")

    map_metric.update(detections, ground_truth_detections)
    eval_metric = map_metric.compute()

    # Access specific attributes
    print(f"mAP@50: {eval_metric.map50}")
    print(f"mAP@50:95: {eval_metric.map50_95}")
    print(f'mAP_scores: {eval_metric.mAP_scores}')
    print(f"AP_per_class: {eval_metric.ap_per_class}")
    print(f'matched classes: {eval_metric.matched_classes}')


    if show_results:
        annotate_and_display11(image, detections, ground_truth_detections)
    if log:
        log_results_to_json(update_batch_id(), image_name, eval_metric.map50, eval_metric.map50_95, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    #if sali:
       # saliance(model, image)
    if eval_metric.map50_95 is None:
        return 0, 0
    return eval_metric.map50_95, eval_metric.map50



if __name__ == "__main__":
    image = "2dark5_jpg.rf.195ff3de514bdb9771d4100d2e00f7ce.jpg"
    image = "09a245b6-498e-4169-9acd-73f09ac4e04b_jpeg_jpg.rf.4a1a07dc1571709c7e7995f48ac6804e.jpg"
    image = "camera_12.png"
    image = "camera_120.png"

    script_dir = get_root_dir()
    model_path = "weights.pt"
    model_path = os.path.join(script_dir, model_path)
    model = YOLO(model_path)
    m = 0
    map_50 = []
    map_95 = []
    for image in os.listdir(os.path.join(script_dir, "data", "3", "images")):
        print(f'Image {m}')
        s, d = evaluate_single_11(image, show_results=False, log=False, sali=True, model = model)
        map_50.append(d)
        map_95.append(s)
        m += 1
        print("--------------------------------------------------")
    #evaluate_single_11(image, show_results=True, log=False, sali=True, model = model)
    avg_50 = np.average(map_50)
    avg_95 = np.average(map_95)
    print(f"Average mAP@50: {avg_50}")
    print(f"Average mAP@50:95: {avg_95}")
