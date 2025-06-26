import numpy as np
import torch
from deep_sort.deep.feature_extractor import Extractor
from deep_sort.deep_sort_folder.tracker import Tracker
from deep_sort.deep_sort_folder.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort_folder.detection import Detection
from deep_sort.deep_sort_folder.preprocessing import non_max_suppression
from deep_sort.deep_sort_folder.tools import xywh_to_xyxy, xyxy_to_xywh


class DeepSort:
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        # Minimum güven skoru (altında kalanlar elenir)
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]

        # Özellik çıkarımı
        features = self.extractor(ori_img, bbox_xywh)
        detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(bbox_xywh, confidences, features)]

        # Non-Maximum Suppression : çakışan kutular filtrelenir
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        
        self.tracker.predict() # Kalman filtresi ile bir sonraki konumu tahmin et
        self.tracker.update(detections) # Yeni tespitlerle eşleştir ve takipleri güncelle , hungarian eşleme

        #takip sonuçları
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # Takip edilen kutuyu xyxy formatına çevir
            bbox = xywh_to_xyxy(track.to_tlwh())
            track_id = track.track_id
            outputs.append(np.array([*bbox, track_id], dtype=np.float32))
        if len(outputs) > 0:
            return np.stack(outputs, axis=0)
        return np.array([])

    def increment_ages(self):
        self.tracker.increment_ages()
