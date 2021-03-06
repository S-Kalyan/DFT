import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow
from GaussianInput import GaussInp1D
from NeuralField import NeuralField1D
import time
from LateralInteraction import LaterealInteraction1D


def make_fig(x):
    plt.plot(x)


# plt.ion()  # enable interactivity
fig = plt.figure()

fieldSize = 100
sigma_exc = 5
sigma_inh = 12.5

inp1 = GaussInp1D(fieldSize, amp=4, mean=20, sigma=5)
inp2 = GaussInp1D(fieldSize, amp=5, mean=60, sigma=5)
dnf1 = NeuralField1D(fieldSize)

lat1 = LaterealInteraction1D(fieldSize, sigma_exc, sigma_inh, amplitudeInh=3, amplitudeExc=3, amplitudeGlobal=0)
x3 = lat1.out
for i in range(1000):
    x1 = inp1.step()
    x2 = inp2.step()
    y = dnf1.step(x1 + x2 + x3)
    x3 = lat1.step(dnf1)

    print(i)
    # print(np.max(y))
    # plt.subplot(2, 1, 1)
    # plt.plot(y)
    # plt.yticks([])
    # plt.subplot(2, 1, 2)
    # plt.plot(x1 + x2)
    # plt.yticks([])
    # plt.show(block=False)
    # plt.pause(0.1)
    # plt.clf()
    # inp1.change_mean(i)
