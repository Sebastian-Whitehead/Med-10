import numpy as np
def xyxy_to_cxcywh(xyxy):
    xyxy = np.array(xyxy)
    x_min, y_min, x_max, y_max = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return np.stack((cx, cy, w, h), axis=1)

def cxcywh_to_xyxy(cxcywh):
    cxcywh = np.array(cxcywh)
    cx, cy, w, h = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return np.stack((x_min, y_min, x_max, y_max), axis=1)

def process_annotations(annotations):
    converted_annotations = []
    class_ids = []
    
    for anno in annotations:
        cxcywh, _, _, class_id, _, _ = anno  # Extract relevant values
        xyxy = cxcywh_to_xyxy(np.array([cxcywh]))[0]  # Convert format
        converted_annotations.append(xyxy)
        class_ids.append(class_id)
    
    return np.array(converted_annotations, dtype=np.float32), np.array(class_ids, dtype=np.int32)
