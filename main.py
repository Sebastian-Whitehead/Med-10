from inference import get_model
import supervision as sv
import cv2
import os
import json
from datetime import datetime

def process_image(image_file):
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
    confidence_values = []
    for i, detection in enumerate(detections.confidence):
        label = f"{detections.class_id[i]} ({detection*100:.2f}%)"  # Convert confidence to percentage
        x1, y1, x2, y2 = detections.xyxy[i]  # Get coordinates for the bounding box
        # Position the label near the top-left corner of the bounding box
        cv2.putText(annotated_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        confidence_values.append(detection)

    # Display annotated image
    sv.plot_image(annotated_image)
    return confidence_values


# Process a batch of images and output a json log file
def process_batch_images(folder_path, batch_number):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_file = os.path.join(folder_path, filename)
            confidence_values = process_image(image_file)
            print(f"Confidence values for {filename}: {confidence_values}")

            log_entry = {
                "batch_number": batch_number,
                "filename": filename,
                "confidence_values": confidence_values
            }
            log_data.append(log_entry)
    
    # Save log data to a JSON file
    json_filename = f"confidence_log{batch_number}_{timestamp}.json"

    #Write log data to JSON file
    with open(json_filename, "w") as json_file:
        json.dump(log_data, json_file, indent = 4)


#batch_images_folder = "batch_images"
#process_batch_images(batch_images_folder, 1)

image_single = "images/4F8A8238.jpg"
process_image(image_single)




