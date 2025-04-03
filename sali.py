from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall
import numpy as np
import os
from PIL import Image
import cv2
from logger import log_results_to_json


def evaluate_single(image_path, annotation_path, show_results = False, log = False):
    #model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")
    #import roboflow
    #rf = roboflow.Roboflow(api_key="r5e027ZfsLmnqRVzqpYa")

    #project = rf.workspace().project("beverage-containers-3atxb/3")

    #model = project.version("1").mode
    #model.download(format='pt', location='.')
    from ultralytics import YOLO
    #CLIENT = InferenceHTTPClient(
    #    api_url="https://detect.roboflow.com",
    #    api_key="r5e027ZfsLmnqRVzqpYa"
    #)
    print("loading weights")
    model_path = "weights.pt"  # Replace with the actual path
    print("weights loaded")
    CLIENT = YOLO(model_path)
    print("client loaded")
    map_metric = MeanAveragePrecision()

    print(f"Evaluating single image: {image_path}")
    #result = model.infer([image_path])[0]
    

    
    
    # Load the image to get its dimensions
    #image = Image.open(image_path)
    image = cv2.imread(image_path)
    results = CLIENT(image)
    for result in results:
        # Extract and print box details
        if result.boxes is not None and len(result.boxes) > 0:
            continue
        else:
            print("No valid detections found.")

    #result = CLIENT.infer(image, model_id="beverage-containers-3atxb/3")
    # Extract bounding boxes, confidence scores, and class labels
    xyxy = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class indices

    # Create detections manually
    detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)

    # Print to verify
    print(f"Detections: {detections}")

    image_width, image_height = image.shape[:2]
    print("")
    # Manually parse the YOLOv8 annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        annotations = []
        for line in lines:
            print(line)
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            x_max = (x_center + width / 2) * image_width
            y_max = (y_center + height / 2) * image_height
            print(x_min, y_min, x_max, y_max)
            annotations.append([x_min, y_min, x_max, y_max, class_id])

    annotations = np.array(annotations)
    print("")
    print(image.shape[:2])
    print(annotations)
    print("")
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

    # Annotate and display the image
    if show_results:
        annotate_and_display(image, detections, ground_truth_detections)
    if log:
        log_results_to_json(batch_id, os.path.dirname(image_path), eval_metric.map50, eval_metric.map50_95, None, None)
    sali(CLIENT, image)
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
        single_tile_size=(image.shape[1], image.shape[0]),
        tile_padding_color=sv.Color.WHITE,
        tile_margin_color=sv.Color.WHITE
    )

    cv2.imshow("Comparison", comparison_image)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()  # Close the window

def update_batch_id():
    global batch_id
    batch_id = datetime.now().strftime("%m%d%H%M%S")

def sali(model_pt, image):
    print("sali")
    

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from ultralytics import YOLO
    from yolo.yolo_cam.eigen_cam import EigenCAM
    from yolo.yolo_cam.utils.image import show_cam_on_image

    

    plt.rcParams["figure.figsize"] = [3.0, 3.0]


    #img = cv2.imread(image)
    #img = cv2.resize(img, (640, 640))
    img = image.copy()
    rgb_img = image.copy()
    img = np.float32(img) / 255

    #model = YOLO(model_pt) 
    model = model_pt
    model = model.cpu()

    target_layers = [model.model.model[-2]]

    cam = EigenCAM(model, target_layers,task='od')

    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.show()

