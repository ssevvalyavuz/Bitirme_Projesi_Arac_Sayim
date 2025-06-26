import numpy as np


def non_max_suppression(boxes, max_overlap, scores=None):
    """NMS: çakışan kutuları eleyerek en güvenilir olanları seçer."""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1] if scores is not None else np.arange(len(boxes))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= max_overlap)[0]
        order = order[inds + 1]

    return keep
