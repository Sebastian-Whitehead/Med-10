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
from utils.saliency import saliance
from ultralytics import YOLO

#Only works with EigenCAM imported

def get_image_anno_path_x(image_name):
    root_dir = get_root_dir()
    image_path = os.path.join(root_dir, "data", "default", "images", image_name)
    base_name, _ = os.path.splitext(image_name)
    annotation_path = os.path.join(root_dir, "data", "default", "labels", f"{base_name}.txt")
    return image_path, annotation_path

def evaluate_single_11(image_name, show_results=False, log=False, sali=False, model_path="weights.pt"):
    script_dir = get_root_dir()
    model_path = os.path.join(script_dir, model_path)
    model = YOLO(model_path)

    map_metric = MeanAveragePrecision()
    image_path, annotation_path = get_image_anno_path_x(image_name)

    image = cv2.imread(image_path)
    results = model(image)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
    else:
        print("No valid detections found.")
        return None

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
    confidence = np.ones(len(class_id))

    ground_truth_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id
    )

    print(f"Detections: {detections}")
    print(f"Ground Truth: {ground_truth_detections}")

    map_metric.update(detections, ground_truth_detections)
    eval_metric = map_metric.compute()

    eval_metric = map_metric.compute()

    # Inspect all attributes
    print("Attributes of eval_metric:")
    print(dir(eval_metric))

    # Access specific attributes
    print(f"mAP@50: {eval_metric.map50}")
    print(f"mAP@50:95: {eval_metric.map50_95}")
    print(f'mAP_scores: {eval_metric.mAP_scores}')
    print(f'mAP_scores: {eval_metric.mAP_scores}')
    print(f"AP_per_class: {eval_metric.ap_per_class}")
    print(f'matched classes: {eval_metric.matched_classes}')


    if show_results:
        annotate_and_display11(image, detections, ground_truth_detections)
    if log:
        log_results_to_json(update_batch_id(), image_name, eval_metric.map50, eval_metric.map50_95, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    if sali:
        saliance(model, image)
    return eval_metric.map50_95



if __name__ == "__main__":
    image = "2dark5_jpg.rf.195ff3de514bdb9771d4100d2e00f7ce.jpg"
    image = "09a245b6-498e-4169-9acd-73f09ac4e04b_jpeg_jpg.rf.4a1a07dc1571709c7e7995f48ac6804e.jpg"
    image = "camera_42.png"
    evaluate_single_11(image, show_results=True, log=False, sali=True)
