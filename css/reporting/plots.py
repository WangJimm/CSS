from __future__ import annotations

import matplotlib.pyplot as plt


def simple_plot(x, xlabel = "Episode", ylabel = "Reward", filename = "tmp123.png"):
    plt.figure()
    plt.plot(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

