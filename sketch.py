import matplotlib.pyplot as plt
import numpy as np


def show_fitness_curve(data, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot([data[i] for i in range(0, len(data))], color="#008855", linewidth=3)
    plt.show()


def show_avg_fitness_curve(data, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    episodes = len(data[0])
    sum_ = np.full([episodes], 0, dtype=float)
    for d in data:
        # plt.plot(d, color="#7FC3AA")
        for i in range(0, episodes):
            sum_[i] += d[i]
    plt.plot([s / len(data) for s in sum_], color="#008855", linewidth=3)
    plt.show()


def show_vs2_fitness_curve(main_data, main_name, sub_data, sub_name,
                           title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    episodes = len(main_data[0])
    main_sum = np.full([episodes], 0, dtype=float)
    for d in main_data:
        plt.plot(d, color="#7FC3AA")
        for i in range(0, episodes):
            main_sum[i] += d[i]
    plt.plot([s / len(main_data) for s in main_sum], color="#008855", label=main_name, linewidth=3)

    sub_sum = np.full([episodes], 0, dtype=float)
    for d in sub_data:
        for i in range(0, episodes):
            sub_sum[i] += d[i]
    plt.plot([s / len(sub_data) for s in sub_sum], color="#880033", label=sub_name, linewidth=3)
    plt.legend()
    plt.show()


def show_vs_multi_fitness_curve(datas, names, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return"):
    colors = ["#008855", "#880033", "#550088", "#885500", "#007788", "#003388",
              "#338800", "#880077", "#881100", "#778800"]

    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for j in range(len(datas)):
        episodes = len(datas[j][0])
        sum_ = np.full([episodes], 0, dtype=float)
        for d in datas[j]:
            for i in range(0, episodes):
                sum_[i] += d[i]
        plt.plot([s / len(datas[j]) for s in sum_], color=colors[j], label=names[j], linewidth=3)
    plt.legend()
    plt.show()
