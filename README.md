# Med-10

git import Eigen_cam into a "yolo" folder. change imports in base_camp.py and eigen_cam.py from:

#base_cam.py
"
from yolo_cam.activations_and_gradients import ActivationsAndGradients
from yolo_cam.utils.svd_on_activations import get_2d_projection
from yolo_cam.utils.image import scale_cam_image
from yolo_cam.utils.model_targets import ClassifierOutputTarget
"
to
"
from yolo.yolo_cam.activations_and_gradients import ActivationsAndGradients
from yolo.yolo_cam.utils.svd_on_activations import get_2d_projection
from yolo.yolo_cam.utils.image import scale_cam_image
from yolo.yolo_cam.utils.model_targets import ClassifierOutputTarget
"
#eigen_cam.py
"
from yolo_cam.base_cam import BaseCAM
from yolo_cam.utils.svd_on_activations import get_2d_projection
"
to
"
from yolo.yolo_cam.base_cam import BaseCAM
from yolo.yolo_cam.utils.svd_on_activations import get_2d_projection
"