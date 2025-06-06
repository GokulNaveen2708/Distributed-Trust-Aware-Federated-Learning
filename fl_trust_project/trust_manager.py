
from collections import defaultdict
import numpy as np

class TrustManager:
    """
    Maintains per‑client trust scores and applies the exponential
    smoothing update described in Eq.(1) of the paper.
    """
    def __init__(self, gamma: float = 0.7, beta: float = 0.2):
        self.gamma = gamma  # smoothing factor
        self.beta = beta    # exclusion threshold
        self.trust = defaultdict(lambda: 1.0)  # init all clients with trust = 1

    def update(self, cid: str, reputation_metric: float):
        self.trust[cid] = self.gamma * self.trust[cid] + (1 - self.gamma) * reputation_metric
        return self.trust[cid]

    def is_trusted(self, cid: str) -> bool:
        return self.trust[cid] >= self.beta

    def get_scores(self):
        return dict(self.trust)
