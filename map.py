from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall
from inference import get_model
import numpy as np
import os
from PIL import Image
from logger import log_results_to_json

batch_id = datetime.now().strftime("%m%d%H%M%S")

def evaluate_batch(batch_path):
    model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")
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

    # Log results to JSON file
    log_results_to_json(batch_id, batch_path, eval_metric.map50, eval_metric.map50_95, pre.precision_at_50, rec.recall_at_50)

    return eval_metric.map50

def evaluate_single(image_path, annotation_path, show_results = False):
    model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")
    map_metric = MeanAveragePrecision()

    print(f"Evaluating single image: {image_path}")
    result = model.infer([image_path])[0]

    detections = sv.Detections.from_inference(result)
    print(f"Detections: {detections}")

    # Load the image to get its dimensions
    image = Image.open(image_path)
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
    print(f"Ground Truth: {ground_truth_detections}")

    map_metric.update(detections, ground_truth_detections)
    eval_metric = map_metric.compute()
    print(f"Mean Average Precision for single image: {eval_metric.map50_95}")

    # Annotate and display the image
    if show_results:
        annotate_and_display(image, detections, ground_truth_detections)

    # Log results to JSON file
    log_results_to_json(batch_id, os.path.dirname(image_path), eval_metric.map50, eval_metric.map50_95, None, None)

    return eval_metric.map50_95

def annotate_and_display(image, detections, ground_truth_detections):
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels_model = [f"Model: {class_id}" for class_id in detections.class_id]
    labels_gt = [f"GT: {class_id}" for class_id in ground_truth_detections.class_id]

    annotated_image_model = image.copy()
    annotated_image_model = box_annotator.annotate(annotated_image_model, detections)
    annotated_image_model = label_annotator.annotate(annotated_image_model, detections, labels_model)

    annotated_image_gt = image.copy()
    annotated_image_gt = box_annotator.annotate(annotated_image_gt, ground_truth_detections)
    annotated_image_gt = label_annotator.annotate(annotated_image_gt, ground_truth_detections, labels_gt)

    # Create a side-by-side comparison
    comparison_image = sv.create_tiles(
        [annotated_image_model, annotated_image_gt],
        grid_size=(1, 2),
        single_tile_size=(image.width, image.height),
        tile_padding_color=sv.Color.WHITE,
        tile_margin_color=sv.Color.WHITE
    )

    comparison_image.show()

def update_batch_id():
    global batch_id
    batch_id = datetime.now().strftime("%m%d%H%M%S")