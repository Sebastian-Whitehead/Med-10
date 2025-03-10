import supervision as sv
from supervision.metrics import MeanAveragePrecision
from inference import get_model
import matplotlib.pyplot as plt
import os

def read_ground_truth(label_file):
    ground_truth = []
    with open(label_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            ground_truth.append((class_id, x_center, y_center, width, height))
    return ground_truth

def evaluate_single(image_file, label_file):
    model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")

    results = model.infer(image_file)[0]
    detections = sv.Detections.from_inference(results)

    ground_truth = read_ground_truth(label_file)
    eval_metric = MeanAveragePrecision().update([detections], [ground_truth]).compute()
    print(f"Detections: {detections}")
    print(f"Mean Average Precision: {eval_metric.map50_95}")
    print(f"Precision: {eval_metric.precision}")
    print(f"Recall: {eval_metric.recall}")

    return eval_metric.map50_95, eval_metric.recall

def evaluate_batch(images_path, labels_path):
    model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")
    map_metric = MeanAveragePrecision()

    for filename in os.listdir(images_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_file = os.path.join(images_path, filename)
            print(f"Evaluating image: {image_file}")

            results = model.infer(image_file)[0]
            detections = sv.Detections.from_inference(results)

            label_file = os.path.join(labels_path, os.path.splitext(filename)[0] + ".txt")
            if os.path.exists(label_file):
                ground_truth = read_ground_truth(label_file)
                map_metric.update([detections], [ground_truth])

    eval_metric = map_metric.compute()
    print(f"Mean Average Precision for batch: {eval_metric.map50_95}")
    print(f"Precision for batch: {eval_metric.precision}")
    print(f"Recall for batch: {eval_metric.recall}")

    return eval_metric.map50_95, eval_metric.recall




single_image = "batch_images/test_set/images/0a20baf628fa9be7_jpg.rf.6ce55e30ca2cac3c5e01b3dacedc7b11.jpg"
single_label = "batch_images/test_set/labels/0a20baf628fa9be7_jpg.rf.6ce55e30ca2cac3c5e01b3dacedc7b11.txt"
evaluate_single(single_image, single_label)

test_images_folder = "batch_images/test_set/images"
test_labels_folder = "batch_images/test_set/labels"
evaluate_batch(test_images_folder, test_labels_folder)

