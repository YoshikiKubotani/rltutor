import numpy as np
from collections import deque


class RandomTutor:
    def __init__(self, n_items, num_offering_questions, seed):
        self.name = "RandomTutor"
        np.random.seed(seed)
        self.n_items = n_items
        self.queue = deque()
        self.queue.extend(
            np.random.randint(0, n_items, num_offering_questions)
        )

    def act(self):
        return self.queue.pop()
