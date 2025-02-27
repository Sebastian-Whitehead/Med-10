from inference import get_model
import supervision as sv
import cv2

# Load image
image_file = "images/wellfamily-facebookJumbo-v2.jpg"
image = cv2.imread(image_file)

# Load model
model = get_model(model_id="beverage-containers-3atxb/3", api_key="r5e027ZfsLmnqRVzqpYa")

# Perform inference
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)

# Annotate image with bounding boxes and labels
bounding_box_annotator = sv.BoxAnnotator()

# Get bounding box coordinates and class labels
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)

# Add confidence percentages to image
for i, detection in enumerate(detections.confidence):
    label = f"{detections.class_id[i]} ({detection*100:.2f}%)"  # Convert confidence to percentage
    x1, y1, x2, y2 = detections.xyxy[i]  # Get coordinates for the bounding box
    # Position the label near the top-left corner of the bounding box
    cv2.putText(annotated_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display annotated image
sv.plot_image(annotated_image)

