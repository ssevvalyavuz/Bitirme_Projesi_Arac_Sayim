import numpy as np

def xywh_to_xyxy(xywh):
    """
    (center_x, center_y, width, height) → (x1, y1, x2, y2)
    """
    x_c, y_c, w, h = map(float, xywh)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return [x1, y1, x2, y2]

def xyxy_to_xywh(xyxy):
    """
    (x1, y1, x2, y2) → (center_x, center_y, width, height)
    """
    x1, y1, x2, y2 = map(float, xyxy)
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2
    y_c = y1 + h / 2
    return [x_c, y_c, w, h]
