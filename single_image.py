import os
import numpy as np
from PIL import Image
import roboflow
from inference import get_model
from supervision.metrics import MeanAveragePrecision
import supervision as sv
from logger import log_results_to_json

def evaluate_single(image_path, annotation_path, show_results = False, log=False):
    model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")
    
    rf = roboflow.Roboflow(api_key="r5e027ZfsLmnqRVzqpYa")

    project = rf.workspace().project("beverage-containers-3atxb/3")

    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="r5e027ZfsLmnqRVzqpYa"
    )
    map_metric = MeanAveragePrecision()

    print(f"Evaluating single image: {image_path}")

    # Load the image to get its dimensions
    image = Image.open(image_path)
    result = CLIENT.infer(image, model_id="beverage-containers-3atxb/3")
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
        annotate_and_display(image, detections, ground_truth_detections)
    if log:
        log_results_to_json(batch_id, os.path.dirname(image_path), eval_metric.map50, eval_metric.map50_95, None, None)
    return eval_metric.map50_95


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    image = "2dark5_jpg.rf.195ff3de514bdb9771d4100d2e00f7ce.jpg"

    base_name, extension = image.rsplit(".", 1)
    txt_image = f"{base_name}.txt"

    single_image_path = os.path.join("batch_images", "test_set", "images", image)
    single_annotation_path = os.path.join("batch_images", "test_set", "labels", txt_image)

    evaluate_single(single_image_path, single_annotation_path, True, False)