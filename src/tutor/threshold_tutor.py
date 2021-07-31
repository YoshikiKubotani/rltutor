import numpy as np

normalize = lambda x: x / x.sum()


class ThresholdTutor:
    def __init__(self, env, threshold):
        self.name = "ThresholdTutor"
        self.threshold = threshold
        self.env = env

    def _next_item(self):
        return np.argmin(
            np.abs(np.array(self.env.get_retention_rate()) - self.threshold)
        )

    def act(self):
        return self._next_item()
