import numpy as np
from collections import deque
from .kalman_filter import KalmanFilter
from .detection import Detection
from .track import Track


class Tracker:
    def __init__(self, distance_metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = distance_metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict()

    def update(self, detections):
        # Eşleştirme (association)
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Eşleşenleri güncelle
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # Eşleşmeyenleri sonlandır veya yaşlandır
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Yeni izleri oluştur
        for detection_idx in unmatched_detections:
            track = Track(
                detections[detection_idx].to_tlwh(),
                self._next_id,
                self.n_init,
                self.max_age,
                detections[detection_idx].feature)
            track.initiate(self.kf)
            self.tracks.append(track)
            self._next_id += 1

        # Sadece aktif track'ler kalmalı
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Özellikleri güncelle
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features = [t.features[-1] for t in self.tracks if t.is_confirmed()]
        self.metric.partial_fit(features, active_targets, active_targets)

    def _match(self, detections):
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_detections = \
            self._min_cost_matching(self.metric.distance, self.max_iou_distance, self.tracks, detections, confirmed_tracks)

        matches_b, unmatched_tracks_b, unmatched_detections = \
            self._min_cost_matching(None, self.max_iou_distance, self.tracks, detections, unconfirmed_tracks, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _min_cost_matching(self, distance_func, max_distance, tracks, detections, track_indices, detection_indices=None):
        if detection_indices is None:
            detection_indices = list(range(len(detections)))

        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], track_indices, detection_indices

        if distance_func is not None:
            features = [detections[i].feature for i in detection_indices]
            targets = [tracks[i].track_id for i in track_indices]
            cost_matrix = distance_func(features, targets)
        else:
            cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

        cost_matrix[cost_matrix > max_distance] = np.inf

        matches, unmatched_tracks, unmatched_detections = \
            self._greedy_match(cost_matrix, track_indices, detection_indices)

        return matches, unmatched_tracks, unmatched_detections

    def _greedy_match(self, cost_matrix, track_indices, detection_indices):
        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_detections = list(detection_indices)

        while cost_matrix.size > 0:
            i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            if cost_matrix[i, j] == np.inf:
                break
            track_idx = track_indices[i]
            detection_idx = detection_indices[j]
            matches.append((track_idx, detection_idx))
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(detection_idx)
            cost_matrix = np.delete(cost_matrix, i, 0)
            cost_matrix = np.delete(cost_matrix, j, 1)
            track_indices.pop(i)
            detection_indices.pop(j)

        return matches, unmatched_tracks, unmatched_detections

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
