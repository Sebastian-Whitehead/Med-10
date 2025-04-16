import os
import numpy as np
from PIL import Image
import roboflow
from inference import get_model
from supervision.metrics import MeanAveragePrecision
from utils.utilities import annotate_and_display8
from utils.utilities import update_batch_id
from utils.utilities import get_root_dir
from utils.utilities import get_image_anno_path
import supervision as sv
from utils.logger import log_results_to_json

def evaluate_single_8(image_name, show_results = False, log=False):
    model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")
    
    map_metric = MeanAveragePrecision()
    image_path, annotation_path = get_image_anno_path(image_name)

    image = Image.open(image_path)
    result = model.infer([image_path])[0]
    print(f"Result: {result}")
    detections = sv.Detections.from_inference(result)
    print(f"Detections: {detections}")

    image_width, image_height = image.size

    # Manually parse the YOLOv8 annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        annotations = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            x_max = (x_center + width / 2) * image_width
            y_max = (y_center + height / 2) * image_height
            annotations.append([x_min, y_min, x_max, y_max, class_id])

    annotations = np.array(annotations)
    xyxy = annotations[:, :4]
    class_id = annotations[:, 4].astype(int)
    confidence = np.ones(len(class_id))  # Assuming confidence is 1 for ground truth

    ground_truth_detections = sv.Detections(
        xyxy=xyxy,
        mask=None,
        confidence=confidence,
        class_id=class_id,
        tracker_id=None,
        data={'class_name': np.array(['class_name_placeholder'] * len(class_id))},  # Replace with actual class names if available
        metadata={}
    )
    #print(f"Ground Truth: {ground_truth_detections}")
    
    map_metric.update(detections, ground_truth_detections)
    eval_metric = map_metric.compute()
    print(f"Mean Average Precision for single image: {eval_metric.map50_95}")

    # Annotate and display the image
    if show_results:
        annotate_and_display8(image, detections, ground_truth_detections)
    if log:
        log_results_to_json(update_batch_id(), image_name, eval_metric.map50, eval_metric.map50_95, None, None)
    return eval_metric.map50_95

if __name__ == "__main__":
    image = "2dark5_jpg.rf.195ff3de514bdb9771d4100d2e00f7ce.jpg"
    image = "09a245b6-498e-4169-9acd-73f09ac4e04b_jpeg_jpg.rf.4a1a07dc1571709c7e7995f48ac6804e.jpg"

    evaluate_single_8(image, True, False)