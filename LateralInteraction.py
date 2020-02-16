import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class LaterealInteraction1D(object):
    def __init__(self, size, sigmaExc, sigmaInh, amplitudeExc=0, amplitudeInh=0, amplitudeGlobal=0, circular=True,
                 normalized=True,
                 cutoffFactor=5):
        self.size = size
        self.sigmaExc = sigmaExc
        self.amplitudeExc = amplitudeExc
        self.sigmaInh = sigmaInh
        self.amplitudeInh = amplitudeInh
        self.amplitudeGlobal = amplitudeGlobal
        self.circular = circular
        self.normalized = normalized
        self.cutoffFactor = cutoffFactor
        self.define_kernel_range()
        self.fullSum = 0
        self.out = np.zeros(self.size)

    def gaussNorm(self, range_x, mu, sigma):
        g = np.exp(-0.5 * (range_x - mu) ** 2 / sigma ** 2)
        g = g / np.sum(g)
        return g

    def conv2(self, v1, v2, m, mode='valid'):
        """
        Two-dimensional convolution of matrix m by vectors v1 and v2

        First convolves each column of 'm' with the vector 'v1'
        and then it convolves each row of the result with the vector 'v2'.

        """
        tmp = np.apply_along_axis(np.convolve, 0, m, v1, mode)
        return np.apply_along_axis(np.convolve, 0, tmp, v2, mode)

    def define_kernel_range(self):
        self.kernelRange = self.cutoffFactor * np.max(
            ((self.amplitudeExc != 0) * self.sigmaExc, (self.amplitudeInh != 0) * self.sigmaInh))
        self.kernelRangeLeft = np.min((np.ceil(self.kernelRange), np.floor((self.size - 1) / 2)))
        self.kernelRangeRight = np.min((np.ceil(self.kernelRange), np.ceil((self.size - 1) / 2)))
        self.extIndex = np.hstack((np.arange(self.size - self.kernelRangeRight, self.size), np.arange(self.size),
                                   np.arange(self.kernelRangeLeft))).astype(np.int64)
        range_x = np.arange(-1 * self.kernelRangeLeft, self.kernelRangeRight + 1)
        self.kernel = self.amplitudeExc * self.gaussNorm(range_x, 0,
                                                         self.sigmaExc) - self.amplitudeInh * self.gaussNorm(range_x, 0,
                                                                                                             self.sigmaInh)
        # plt.plot(self.kernel)
        # plt.show()

        # pass

    def step(self, dnf):
        input = dnf.out
        self.fullSum = np.sum(input)
        self.out = self.conv2(np.array([1]), self.kernel, input[self.extIndex],
                              'valid') + self.amplitudeGlobal * self.fullSum

        return self.out
