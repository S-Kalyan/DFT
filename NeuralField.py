import numpy as np
import scipy


class NeuralField1D(object):
    def __init__(self, size):
        self.size = size
        self.tau = 20
        self.h = -5
        self.beta = 4
        self.deltaT = 1
        self.activation = np.zeros(self.size) + self.h
        self.reset_field()
        self.out = self.sigmoid(self.activation, self.beta, 0)

    def sigmoid(self, x, beta, x0):
        out = 1 / (1 + np.exp(-beta * (x - x0)))
        return out

    def reset_field(self):
        pass
        # self.activaton = np.zeros(self.size) + self.h * np.ones(self.size)

    def step(self, inputs):
        self.activation = self.activation + (self.deltaT / self.tau) * (- 1 * self.activation + self.h + inputs)
        self.out = self.sigmoid(self.activation, self.beta, 0)
        return self.out
