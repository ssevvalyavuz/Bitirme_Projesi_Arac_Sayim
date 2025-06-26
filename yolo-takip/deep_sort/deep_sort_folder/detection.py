import numpy as np


class Detection:
    """
    Bir tespiti temsil eder:
    - bbox: (center_x, center_y, width, height)
    - confidence: YOLO modelinden gelen güven skoru
    - feature: Deep feature (appearance) vektörü
    """
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Sol üst ve sağ alt köşe: (x1, y1, x2, y2)"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xywh(self):
        """(x_center, y_center, width, height)"""
        return self.tlwh.copy()

    def to_tlwh(self):
        """(top left x, top left y, width, height)"""
        return self.tlwh.copy()
