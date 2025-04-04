import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from yolo.yolo_cam.eigen_cam import EigenCAM
from yolo.yolo_cam.utils.image import show_cam_on_image

def saliance(model_pt, image):
    plt.rcParams["figure.figsize"] = [3.0, 3.0]

    img = image.copy()
    rgb_img = image.copy()
    img = np.float32(img) / 255

    model = model_pt
    model = model.cpu()

    target_layers = [model.model.model[-2]]

    cam = EigenCAM(model, target_layers,task='od')

    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.show()

