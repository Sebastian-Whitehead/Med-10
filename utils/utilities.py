import os
import cv2
import supervision as sv
import datetime

def get_root_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    root_dir = script_dir
    while not os.path.exists(os.path.join(root_dir, "main.py")):
        parent_dir = os.path.dirname(root_dir)
        if parent_dir == root_dir: 
            raise FileNotFoundError("Root directory with 'main.py' not found.")
        root_dir = parent_dir
    return root_dir

def get_image_anno_path(image_name):
    root_dir = get_root_dir()
    image_path = os.path.join(root_dir, "batch_images", "test_set", "images", image_name)
    base_name, _ = os.path.splitext(image_name)
    annotation_path = os.path.join(root_dir, "batch_images", "test_set", "labels", f"{base_name}.txt")
    return image_path, annotation_path

def annotate_and_display11(image, detections, ground_truth_detections):
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels_model = [f"Model: {class_id}" for class_id in detections.class_id]
    labels_gt = [f"GT: {class_id}" for class_id in ground_truth_detections.class_id]

    annotated_image_model = box_annotator.annotate(image.copy(), detections)
    annotated_image_model = label_annotator.annotate(annotated_image_model, detections, labels_model)

    annotated_image_gt = box_annotator.annotate(image.copy(), ground_truth_detections)
    annotated_image_gt = label_annotator.annotate(annotated_image_gt, ground_truth_detections, labels_gt)
    print("----------------------------")
    print(detections)
    print(ground_truth_detections)
    print("----------------------------")

    # Create a top-to-bottom comparison
    comparison_image = sv.create_tiles(
        [annotated_image_model, annotated_image_gt],
        grid_size=(2, 1), 
        single_tile_size=(image.shape[1], image.shape[0]),
        tile_padding_color=sv.Color.WHITE,
        tile_margin_color=sv.Color.WHITE
    )

    comparison_image = cv2.resize(comparison_image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

    cv2.imshow("Comparison", comparison_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def annotate_and_display8(image, detections, ground_truth_detections):
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

    # Resize the comparison image to make it smaller
    comparison_image = cv2.resize(comparison_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imshow("Comparison", comparison_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def update_batch_id():
    batch_id = datetime.datetime.now().strftime("%m%d%H%M%S")
    return batch_id
