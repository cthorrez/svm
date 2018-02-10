import matplotlib
matplotlib.use('Agg')
import numpy as np

import math

import matplotlib.pyplot as plt

if __name__ == "__main__":
    SGD = np.genfromtxt('SGD_cost.csv', delimiter=',', dtype=float)
    batch = np.genfromtxt('batch_cost.csv', delimiter=',', dtype=float)


    plt.plot(SGD[:,0],SGD[:,1], 'r', label='SGD (total time: 41 sec)')
    plt.plot(batch[:,0],batch[:,1], 'b', label='BGD (total time: 190 sec)')
    plt.xlabel("number of iterations")
    plt.ylabel("cost")
    plt.legend()
    plt.savefig("p1_plot.png")
