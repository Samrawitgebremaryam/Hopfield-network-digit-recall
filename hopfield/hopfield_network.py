import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=10):
        output = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                output[i] = 1 if np.dot(self.weights[i], output) >= 0 else -1
        return output
