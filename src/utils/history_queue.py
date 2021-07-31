class OurQueue:
    """
    A queue for counting efficiently the number of events within time windows.
    Complexity:
        All operators in amortized O(W) time where W is the number of windows.

    From JJ's KTM repository: https://github.com/jilljenn/ktm.
    """

    def __init__(self):
        self.queue = []
        self.window_lengths = [
            3600 * 24 * 30,  # 2592000 (30 days[s])
            3600 * 24 * 7,  # 604800   (7 days[s])
            3600 * 24,  # 86400    (24 hours[s])
            3600,  # 3600     (1 hour[s])
        ]
        self.cursors = [0] * len(self.window_lengths)
        self.counters = [0] * (len(self.window_lengths) + 1)

    def __len__(self):
        return len(self.queue)

    def get_counters(self, t):
        self.update_cursors(t)
        self.counters = [len(self.queue)] + [
            len(self.queue) - cursor for cursor in self.cursors
        ]
        return self.counters

    def push(self, time):
        self.queue.append(time)

    def update_cursors(self, t):
        for pos, length in enumerate(self.window_lengths):
            while (
                self.cursors[pos] < len(self.queue)
                and t - self.queue[self.cursors[pos]] >= length
            ):
                self.cursors[pos] += 1
