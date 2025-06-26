import numpy as np
from scipy.spatial import distance


class NearestNeighborDistanceMetric:
    def __init__(self, metric, matching_threshold, budget=None):
        if metric != "cosine":
            raise ValueError("Sadece cosine mesafesi destekleniyor.")
        self.metric = metric
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]

        self.samples = {k: self.samples[k] for k in active_targets if k in self.samples}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))

        for i, target in enumerate(targets):
            if target not in self.samples:
                cost_matrix[i, :] = np.inf
                continue
            cost_matrix[i, :] = self._cosine_distance(features, self.samples[target])
        return cost_matrix

    @staticmethod
    def _cosine_distance(a, b):
        if len(b) == 0:
            return np.zeros(len(a))
        a = np.asarray(a)
        b = np.asarray(b)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        sim = np.dot(a, b.T) / (a_norm * b_norm.T + 1e-6)
        return 1. - np.max(sim, axis=1)
