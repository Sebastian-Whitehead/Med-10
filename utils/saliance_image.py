import sys
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
sys.path.append(".")
from utils.utilities import get_root_dir, get_image_anno_path

### ATTEMPT AT DOING SMTH SMARTS 'massive fail'

def preprocess_image(image_path, device):
    """
    Preprocesses the image for the model.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(image).permute(2, 0, 1).float()
    img = img / 255.0
    img = img.unsqueeze(0).to(device)
    return image, img

def generate_gradcam(model, image_path, target_layer_name="model.23.cv3.1.0.0.conv", class_idx=None):
    """
    Generates Grad-CAM for the given image and model.
    """
    # Find the target layer
    target_layer = None
    for name, module in model.model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"Layer {target_layer_name} not found in the model")

    # Preprocess the image
    original_image, img = preprocess_image(image_path, model.device)
    img.requires_grad = True

    # Add a small perturbation to the input
    img = img + torch.randn_like(img) * 1e-5

    # Store activations and gradients
    activations = []
    gradients = []

    # Define hooks
    def forward_hook(module, input, output):
        print("Forward hook triggered")  # Debug: Confirm hook is triggered
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        print("Backward hook triggered")  # Debug: Confirm hook is triggered
        gradients.append(grad_out[0])

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass with gradients enabled
    model.model.eval()
    outputs = model.model(img)  # Use the raw model outputs, not post-processed outputs

    # Debug: Inspect outputs
    print(f"Outputs type: {type(outputs)}")
    if isinstance(outputs, tuple):
        print(f"Outputs length: {len(outputs)}")
        print(f"Outputs[0] type: {type(outputs[0])}")

    # Use the correct element from outputs
    raw_predictions = outputs[0]  # Adjust this based on the debug output

    # Use the raw outputs to compute the score
    if class_idx is None:
        class_idx = 0  # Default to the first class if not specified
    score = raw_predictions[0, class_idx].sum()  # Reduce to a scalar
    print(f"Score requires_grad: {score.requires_grad}")  # Debug: Check if score is part of the graph

    # Backward pass
    model.zero_grad()
    score.backward()

    # Debug: Check if gradients are populated
    if not gradients:
        raise RuntimeError("Backward hook was not triggered. Check the target layer.")
    print(f"Gradients shape: {gradients[0].shape}, Gradients sum: {torch.sum(gradients[0])}")

    # Compute Grad-CAM
    activations = activations[0]
    gradients = gradients[0]

    # Debug: Check activations and gradients
    print(f"Activations min: {torch.min(activations)}, max: {torch.max(activations)}")
    print(f"Gradients min: {torch.min(gradients)}, max: {torch.max(gradients)}")

    # Compute weights
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Current method
    # Alternative: weights = torch.sum(gradients, dim=(2, 3), keepdim=True)

    # Compute CAM
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = torch.relu(cam)

    # Debug: Check CAM values before normalization
    cam_np = cam.squeeze().cpu().detach().numpy()
    print(f"CAM min (before normalization): {np.min(cam_np)}, max: {np.max(cam_np)}")

    # Resize CAM to original image size
    cam = cam.squeeze().cpu().detach().numpy()
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    # Normalize CAM
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    # Debug: Check CAM values after normalization
    print(f"CAM min (after normalization): {np.min(cam)}, max: {np.max(cam)}")

    # Convert to heatmap
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert original image to RGB (if needed)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on the original image
    overlaid = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return original_image, heatmap, overlaid

def saliance():
    """
    Main function to generate salience maps.
    """
    script_dir = get_root_dir()
    model_path = os.path.join(script_dir, "weights.pt")
    model = YOLO(model_path)

    image_name = "2dark5_jpg.rf.195ff3de514bdb9771d4100d2e00f7ce.jpg"
    image_path, annotation_path = get_image_anno_path(image_name)
    target_layer_name = "model.23.cv3.1.0.0.conv"  # Adjust this based on your model architecture
    #target_layer_name = "model.23.cv3.1"  # Adjust this based on your model architecture

    original_image, heatmap, overlaid = generate_gradcam(model, image_path, target_layer_name)

    # Display the results
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    saliance()