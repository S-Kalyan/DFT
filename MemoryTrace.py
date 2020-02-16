import numpy as np


class MemoryTrace1D(object):
    def __init__(self, fieldsize):
        self.size = fieldsize
        self.tauBuild = 100
        self.tauDecay = 1000
        self.threshold = 0.5
        self.activeRegions = np.zeros(self.size)
        self.output = np.zeros(self.size)

    def step(self, input):
        self.activeRegions = 1 * (input > self.threshold)
        if any(self.activeRegions):
            self.output = self.output + (1 / self.tauBuild) * (
                    -self.output + input) * self.activeRegions + (1 / self.tauDecay) * (
                              -self.output) * 1 * (1 - self.activeRegions)
        return self.output
