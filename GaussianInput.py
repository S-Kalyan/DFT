import numpy as np
import matplotlib.pyplot as plt


class GaussInp1D(object):
    def __init__(self, size, amp, mean, sigma, mode="min"):
        self.mean = mean
        self.amp = amp
        self.size = size
        self.sigma = sigma
        self.mode = mode
        self.x = self.gauss(np.arange(size), amp, sigma, mean, mode)

    def update_input(self):
        self.x = self.gauss(np.arange(self.size), self.amp, self.sigma, self.mean, self.mode)

    @staticmethod
    def gauss(range_x, amp, sigma, mean, mode):
        l = np.abs(range_x[-1] - 2 * range_x[0] + range_x[1])
        m = np.min(range_x)
        mu_shifted = np.mod(mean - m, l) + m
        if sigma == 0:
            g = 1 * (range_x == mu_shifted)
        else:
            d = np.abs(range_x - mu_shifted)

            if mode == "min":
                dld = np.stack((d, l - d))
                g = np.exp(-0.5 * np.min(dld, 0) ** 2 / sigma ** 2)
            else:
                g = np.exp(-0.5 * d ** 2 / sigma ** 2) + np.exp(-0.5 * (l - d) ** 2 / sigma ** 2)

        return amp * g

    def change_mean(self, mean):
        self.mean = mean
        self.update_input()

    def step(self):
        return self.x
