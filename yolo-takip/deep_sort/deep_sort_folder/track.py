import numpy as np


class Track:
    def __init__(self, tlwh, track_id, n_init, max_age, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.features = [feature]
        self._n_init = n_init
        self._max_age = max_age

        self.state = 'Tentative'

    def to_tlwh(self):
        return self.tlwh.copy()

    def predict(self):
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.tlwh = detection.to_tlwh()
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0

        if self.state == 'Tentative' and self.hits >= self._n_init:
            self.state = 'Confirmed'

    def mark_missed(self):
        if self.state == 'Tentative':
            self.state = 'Deleted'
        elif self.time_since_update > self._max_age:
            self.state = 'Deleted'

    def is_tentative(self):
        return self.state == 'Tentative'

    def is_confirmed(self):
        return self.state == 'Confirmed'

    def is_deleted(self):
        return self.state == 'Deleted'

    def initiate(self, kf):
        pass  # Kalman filtre başlatma boş çünkü sadeleştirilmiş versiyon
