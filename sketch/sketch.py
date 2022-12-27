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


def show_avg_tictactoe(data, title="Fitness Curve", subtitle="", x_label="episodes", y_label="return",
                       player_flag=1, opponent_flag=2):
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    episodes = len(data[0])
    player = np.full([episodes], 0, dtype=float)
    draws = np.full([episodes], 0, dtype=float)
    opponent = np.full([episodes], 0, dtype=float)
    for d in data:
        # plt.plot(d, color="#7FC3AA")
        for i in range(0, episodes):
            if d[i] == player_flag:
                player[i] += 1
            elif d[i] == opponent_flag:
                opponent[i] += 1
            else:
                draws[i] += 1

    plt.plot([s / len(data) for s in draws], color="#888888", linewidth=3, label="Draws")
    plt.plot([s / len(data) for s in opponent], color="#880033", linewidth=3, label="Opponent")
    plt.plot([s / len(data) for s in player], color="#008855", linewidth=3, label="Player")

    plt.legend()
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
        for i in range(0, episodes):
            main_sum[i] += d[i]

    sub_sum = np.full([episodes], 0, dtype=float)
    for d in sub_data:
        for i in range(0, episodes):
            sub_sum[i] += d[i]

    plt.plot([s / len(sub_data) for s in sub_sum], color="#880033", label=sub_name, linewidth=3)
    plt.plot([s / len(main_data) for s in main_sum], color="#008855", label=main_name, linewidth=3)
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


def execute_function(function, env=None, n=None, episodes=None, gamma=None, epsilon=None, alpha=None,
                     epsilon_decay=None, updates=False, prev_state=None):
    nn = n is None
    epin = episodes is None
    gn = gamma is None
    en = epsilon is None
    an = alpha is None
    edn = epsilon_decay is None
    ps = prev_state is None

    if ps:
        if nn:
            if epin:
                if gn:
                    if en:
                        if an:
                            if edn:
                                return function(env, updates=updates)
                            else:
                                return function(env, epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, alpha=alpha, updates=updates)
                            else:
                                return function(env, alpha=alpha, epsilon_decay=epsilon_decay, updates=updates)
                    else:
                        if an:
                            if edn:
                                return function(env, epsilon=epsilon, updates=updates)
                            else:
                                return function(env, epsilon=epsilon, epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, epsilon=epsilon, alpha=alpha, updates=updates)
                            else:
                                return function(env, epsilon=epsilon, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates)
                else:
                    if en:
                        if an:
                            if edn:
                                return function(env, gamma=gamma, updates=updates)
                            else:
                                return function(env, gamma=gamma, epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, gamma=gamma, alpha=alpha, updates=updates)
                            else:
                                return function(env, gamma=gamma, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates)
                    else:
                        if an:
                            if edn:
                                return function(env, gamma=gamma, epsilon=epsilon, updates=updates)
                            else:
                                return function(env, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                                updates=updates)
                        else:
                            if edn:
                                return function(env, gamma=gamma, epsilon=epsilon, alpha=alpha, updates=updates)
                            else:
                                return function(env, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates)
            else:
                if gn:
                    if en:
                        if an:
                            if edn:
                                return function(env, episodes=episodes, updates=updates)
                            else:
                                return function(env, episodes=episodes, epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, episodes=episodes, alpha=alpha, updates=updates)
                            else:
                                return function(env, episodes=episodes, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates)
                    else:
                        if an:
                            if edn:
                                return function(env, episodes=episodes, epsilon=epsilon, updates=updates)
                            else:
                                return function(env, episodes=episodes, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                                updates=updates)
                        else:
                            if edn:
                                return function(env, episodes=episodes, epsilon=epsilon, alpha=alpha, updates=updates)
                            else:
                                return function(env, episodes=episodes, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates)
                else:
                    if en:
                        if an:
                            if edn:
                                return function(env, episodes=episodes, gamma=gamma, updates=updates)
                            else:
                                return function(env, episodes=episodes, gamma=gamma, epsilon_decay=epsilon_decay,
                                                updates=updates)
                        else:
                            if edn:
                                return function(env, episodes=episodes, gamma=gamma, alpha=alpha, updates=updates)
                            else:
                                return function(env, episodes=episodes, gamma=gamma, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates)
                    else:
                        if an:
                            if edn:
                                return function(env, episodes=episodes, gamma=gamma, epsilon=epsilon, updates=updates)
                            else:
                                return function(env, episodes=episodes, gamma=gamma, epsilon=epsilon,
                                                epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                updates=updates)
                            else:
                                return function(env, episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates)
        else:
            if epin:
                if gn:
                    if en:
                        if an:
                            if edn:
                                return function(env, n=n, updates=updates)
                            else:
                                return function(env, n=n, epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, n=n, alpha=alpha, updates=updates)
                            else:
                                return function(env, n=n, alpha=alpha, epsilon_decay=epsilon_decay, updates=updates)
                    else:
                        if an:
                            if edn:
                                return function(env, n=n, epsilon=epsilon, updates=updates)
                            else:
                                return function(env, n=n, epsilon=epsilon, epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, n=n, epsilon=epsilon, alpha=alpha, updates=updates)
                            else:
                                return function(env, n=n, epsilon=epsilon, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates)
                else:
                    if en:
                        if an:
                            if edn:
                                return function(env, n=n, gamma=gamma, updates=updates)
                            else:
                                return function(env, n=n, gamma=gamma, epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, n=n, gamma=gamma, alpha=alpha, updates=updates)
                            else:
                                return function(env, n=n, gamma=gamma, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates)
                    else:
                        if an:
                            if edn:
                                return function(env, n=n, gamma=gamma, epsilon=epsilon, updates=updates)
                            else:
                                return function(env, n=n, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                                updates=updates)
                        else:
                            if edn:
                                return function(env, n=n, gamma=gamma, epsilon=epsilon, alpha=alpha, updates=updates)
                            else:
                                return function(env, n=n, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates)
            else:
                if gn:
                    if en:
                        if an:
                            if edn:
                                return function(env, n=n, episodes=episodes, updates=updates)
                            else:
                                return function(env, n=n, episodes=episodes, epsilon_decay=epsilon_decay,
                                                updates=updates)
                        else:
                            if edn:
                                return function(env, n=n, episodes=episodes, alpha=alpha, updates=updates)
                            else:
                                return function(env, n=n, episodes=episodes, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates)
                    else:
                        if an:
                            if edn:
                                return function(env, n=n, episodes=episodes, epsilon=epsilon, updates=updates)
                            else:
                                return function(env, n=n, episodes=episodes, epsilon=epsilon,
                                                epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, n=n, episodes=episodes, epsilon=epsilon, alpha=alpha,
                                                updates=updates)
                            else:
                                return function(env, n=n, episodes=episodes, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates)
                else:
                    if en:
                        if an:
                            if edn:
                                return function(env, n=n, episodes=episodes, gamma=gamma, updates=updates)
                            else:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon_decay=epsilon_decay,
                                                updates=updates)
                        else:
                            if edn:
                                return function(env, n=n, episodes=episodes, gamma=gamma, alpha=alpha, updates=updates)
                            else:
                                return function(env, n=n, episodes=episodes, gamma=gamma, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates)
                    else:
                        if an:
                            if edn:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon,
                                                updates=updates)
                            else:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon,
                                                epsilon_decay=epsilon_decay, updates=updates)
                        else:
                            if edn:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                updates=updates)
                            else:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates)
    else:
        if nn:
            if epin:
                if gn:
                    if en:
                        if an:
                            if edn:
                                return function(env, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, alpha=alpha, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, alpha=alpha, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                    else:
                        if an:
                            if edn:
                                return function(env, epsilon=epsilon, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, epsilon=epsilon, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, epsilon=epsilon, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, epsilon=epsilon, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                else:
                    if en:
                        if an:
                            if edn:
                                return function(env, gamma=gamma, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, gamma=gamma, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, gamma=gamma, alpha=alpha, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, gamma=gamma, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                    else:
                        if an:
                            if edn:
                                return function(env, gamma=gamma, epsilon=epsilon, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, gamma=gamma, epsilon=epsilon, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
            else:
                if gn:
                    if en:
                        if an:
                            if edn:
                                return function(env, episodes=episodes, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, episodes=episodes, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, episodes=episodes, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, episodes=episodes, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                    else:
                        if an:
                            if edn:
                                return function(env, episodes=episodes, epsilon=epsilon, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, episodes=episodes, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, episodes=episodes, epsilon=epsilon, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, episodes=episodes, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
                else:
                    if en:
                        if an:
                            if edn:
                                return function(env, episodes=episodes, gamma=gamma, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, episodes=episodes, gamma=gamma, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, episodes=episodes, gamma=gamma, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, episodes=episodes, gamma=gamma, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
                    else:
                        if an:
                            if edn:
                                return function(env, episodes=episodes, gamma=gamma, epsilon=epsilon, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, episodes=episodes, gamma=gamma, epsilon=epsilon,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                updates=updates, prev_state=prev_state)
                            else:
                                return function(env, episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
        else:
            if epin:
                if gn:
                    if en:
                        if an:
                            if edn:
                                return function(env, n=n, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, n=n, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, n=n, alpha=alpha, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, n=n, alpha=alpha, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                    else:
                        if an:
                            if edn:
                                return function(env, n=n, epsilon=epsilon, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, n=n, epsilon=epsilon, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, n=n, epsilon=epsilon, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, n=n, epsilon=epsilon, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                else:
                    if en:
                        if an:
                            if edn:
                                return function(env, n=n, gamma=gamma, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, n=n, gamma=gamma, epsilon_decay=epsilon_decay, updates=updates,
                                                prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, n=n, gamma=gamma, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, n=n, gamma=gamma, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                    else:
                        if an:
                            if edn:
                                return function(env, n=n, gamma=gamma, epsilon=epsilon, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, n=n, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, n=n, gamma=gamma, epsilon=epsilon, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, n=n, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
            else:
                if gn:
                    if en:
                        if an:
                            if edn:
                                return function(env, n=n, episodes=episodes, updates=updates, prev_state=prev_state)
                            else:
                                return function(env, n=n, episodes=episodes, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, n=n, episodes=episodes, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, n=n, episodes=episodes, alpha=alpha, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                    else:
                        if an:
                            if edn:
                                return function(env, n=n, episodes=episodes, epsilon=epsilon, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, n=n, episodes=episodes, epsilon=epsilon,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, n=n, episodes=episodes, epsilon=epsilon, alpha=alpha,
                                                updates=updates, prev_state=prev_state)
                            else:
                                return function(env, n=n, episodes=episodes, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
                else:
                    if en:
                        if an:
                            if edn:
                                return function(env, n=n, episodes=episodes, gamma=gamma, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon_decay=epsilon_decay,
                                                updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, n=n, episodes=episodes, gamma=gamma, alpha=alpha, updates=updates,
                                                prev_state=prev_state)
                            else:
                                return function(env, n=n, episodes=episodes, gamma=gamma, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
                    else:
                        if an:
                            if edn:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon,
                                                updates=updates, prev_state=prev_state)
                            else:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)
                        else:
                            if edn:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                updates=updates, prev_state=prev_state)
                            else:
                                return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                epsilon_decay=epsilon_decay, updates=updates, prev_state=prev_state)