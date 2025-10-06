import numpy as np


def reservoir_sample(instream, k):
    # reservoir sample algorithm l
    reservoir = list()
    j = 0
    try:
        while j < k:
            reservoir.append(next(instream))
            j += 1
    except StopIteration:
        return reservoir

    w = np.exp(np.log(np.random.random()) / k)
    i = j
    for j, element in enumerate(instream, start=j):
        if j == i:
            reservoir[np.random.randint(0, k)] = element
            i += (np.floor(np.log(np.random.random()) / np.log(1 - w))) + 1
            w *= np.exp(np.log(np.random.random()) / k)
        else:
            pass
    return reservoir


class AlgorithmL:
    # ported from https://richardstartin.github.io/posts/reservoir-sampling#algorithm-l
    def __init__(self, k):
        self.reservoir = list()
        self.k = k
        self.next = k
        self.counter = 0
        self.w = np.exp(np.log(np.random.random()) / k)
        self.skip()

    def add(self, item):
        if self.counter < self.k:
            self.reservoir.append(item)
        else:
            if self.counter == self.next:
                self.reservoir[np.random.randint(0, self.k)] = item
                self.skip()
        self.counter += 1

    def skip(self):
        self.next += (np.floor(
            np.log(np.random.random()) / np.log(1 - self.w))) + 1
        self.w *= np.exp(np.log(np.random.random()) / self.k)
