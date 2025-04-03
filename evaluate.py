from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision
import numpy as np
import os
import cv2
from logger import log_results_to_json
from ultralytics import YOLO
from yolo.yolo_cam.eigen_cam import EigenCAM
from yolo.yolo_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

#Only works with EigenCAM imported

def evaluate_single(image_path, annotation_path, show_results=False, log=False):
    model_path = "weights.pt" 
    CLIENT = YOLO(model_path)

    map_metric = MeanAveragePrecision()

    image = cv2.imread(image_path)
    results = CLIENT(image)

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

    if show_results:
        annotate_and_display(image, detections, ground_truth_detections)
    if log:
        log_results_to_json(datetime.now().strftime("%m%d%H%M%S"), os.path.dirname(image_path), eval_metric.map50, eval_metric.map50_95, None, None)

    sali(CLIENT, image)
    return eval_metric.map50_95


def annotate_and_display(image, detections, ground_truth_detections):
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels_model = [f"Model: {class_id}" for class_id in detections.class_id]
    labels_gt = [f"GT: {class_id}" for class_id in ground_truth_detections.class_id]

    annotated_image_model = box_annotator.annotate(image.copy(), detections)
    annotated_image_model = label_annotator.annotate(annotated_image_model, detections, labels_model)

    annotated_image_gt = box_annotator.annotate(image.copy(), ground_truth_detections)
    annotated_image_gt = label_annotator.annotate(annotated_image_gt, ground_truth_detections, labels_gt)

    comparison_image = sv.create_tiles(
        [annotated_image_model, annotated_image_gt],
        grid_size=(1, 2),
        single_tile_size=(image.shape[1], image.shape[0]),
        tile_padding_color=sv.Color.WHITE,
        tile_margin_color=sv.Color.WHITE
    )

    cv2.imshow("Comparison", comparison_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sali(model, image):
    plt.rcParams["figure.figsize"] = [3.0, 3.0]

    img = np.float32(image) / 255
    target_layers = [model.model.model[-2]]
    cam = EigenCAM(model, target_layers, task='od')

    grayscale_cam = cam(image)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.show()