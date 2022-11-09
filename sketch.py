import matplotlib.pyplot as plt
import numpy as np


def show_fitness_curve(data, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot([data[i] for i in range(0, len(data))], color="#008855")
    plt.show()


def show_avg_fitness_curve(data, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    episodes = len(data[0])
    sum_ = np.full([episodes], 0, dtype=float)
    for d in data:
        plt.plot(d, color="#7FC3AA")
        for i in range(0, episodes):
            sum_[i] += d[i]
    plt.plot([s / len(data) for s in sum_], color="#008855")
    plt.show()

