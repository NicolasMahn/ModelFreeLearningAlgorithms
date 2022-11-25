import monte_carlo_algorithm
import sketch
import quality_algorithm
import value_algorithm
import environemnt
import time


def main():
    cliff = environemnt.Cliff()
    labyrinth = environemnt.Labyrinth()
    tictactoe = environemnt.TicTacToe(1)

    # vs_multi(cliff, 100, [quality_algorithm.sarsa, quality_algorithm.q_learning],
    #          ["SARSA", "Q-Learning"],
    #          [None, None], "Cliff Walking")

    # vs_multi(labyrinth, 1000, [value_algorithm.td_0, value_algorithm.td_n, value_algorithm.td_n, value_algorithm.td_n,
    #                           value_algorithm.td_n, value_algorithm.td_n, monte_carlo_algorithm.monte_carlo],
    #         ["1-step TD", "2-step TD", "4-step TD", "8-step TD", "16-step TD", "32-step TD", "MC"],
    #         [None, 2, 4, 8, 16, 32, None], "Comparing TD Algorithms")

    # vs_multi(labyrinth, 1000, [monte_carlo_algorithm.monte_carlo, monte_carlo_algorithm.monte_carlo_constant_alpha],
    #         ["classical MC", "MC const alpha 0.5"],
    #         [None, 2, 4, 8, 16, 32, None], "Comparing MC Algorithms" episodes_list, gamma_list, epsilon_list, alpha_list, epsilon_decay_list)

    # vs_multi(labyrinth, 1000, [value_algorithm.td_0, monte_carlo_algorithm.monte_carlo], ["TD(0)", "MC"], "TD(0) compared to MC")

    vs_multi_with_par(labyrinth, 1000, [quality_algorithm.sarsa, value_algorithm.td_0], n_list=[None,None], episodes_list=[None,None], gamma_list=[None,None], epsilon_list=[None,None], alpha_list=[0.1, 0.1], epsilon_decay_list=[None,None,], function_names=["SARSA", "TD(0)"], title="SARSA compared to TD(0)")

    #vs_multi_with_par(labyrinth, 100, functions=[quality_algorithm.sarsa, quality_algorithm.sarsa, quality_algorithm.sarsa, value_algorithm.td_0, value_algorithm.td_n, quality_algorithm.q_learning], n_list=[None,None,None,None,2,None], episodes_list=[None,None,None,None,None,None], gamma_list=[None,None,None,None,None,None], epsilon_list=[None,None,None,None,None,None], alpha_list=[0.5, 0.1, 0.01, 0.01, 0.01, 0.01], epsilon_decay_list=[None,None,None, None, None,None], function_names=["alpha 0.5", "alpha 0.1", "alpha 0.01", "TD(0)", "2-step TD", "Q"], title="SARSA alpha test")

    # vs2(labyrinth, 1000, 500, value_algorithm.td_0, "TD(0)",
    #    monte_carlo_algorithm.monte_carlo, "MC")
    # execute_td_0(labyrinth)
    # execute_q_learning(labyrinth)
    # avg_q_learning(labyrinth, 1000)
    # execute_monte_carlo(labyrinth)
    # avg_monte_carlo(labyrinth, 1000)
    # avg_td_0(labyrinth, 100)

    # execute_q_learning(cliff)
    # execute_sarsa(cliff)
    # avg_sarsa(cliff, 100)
    # avg_q_learning(tictactoe, 100)


def vs2(env, number_of_iterations, number_of_episodes, main_, main_name, sub, sub_name):
    fitness_curves_main = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = main_(env, episodes=number_of_episodes)
        fitness_curves_main.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time of {main_name} is: {execution_time / number_of_iterations}")

    fitness_curves_sub = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = sub(env, episodes=number_of_episodes)
        fitness_curves_sub.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time of {sub_name} is: {execution_time / number_of_iterations}")

    sketch.show_vs2_fitness_curve(fitness_curves_main, main_name, fitness_curves_sub, sub_name,
                                  title=f"Fitness Curve comparing {main_name} with {sub_name}",
                                  subtitle=f"average of {number_of_iterations} iterations")


def vs_multi_with_par(env, number_of_iterations, functions, n_list, episodes_list, gamma_list,
                      epsilon_list, alpha_list, epsilon_decay_list, function_names, title="title"):
    fitness_curves_list = list()
    for j in range(len(functions)):
        fitness_curves = list()
        start_time = time.perf_counter()
        for i in range(0, number_of_iterations):
            _, fitness_curve = execute_function(functions[j], env, n_list[j], episodes_list[j],
                                                gamma_list[j], epsilon_list[j], alpha_list[j], epsilon_decay_list[j])
            fitness_curves.append(fitness_curve)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"The execution time of {function_names[j]} is: {execution_time / number_of_iterations}")
        fitness_curves_list.append(fitness_curves)

    sketch.show_vs_multi_fitness_curve(fitness_curves_list, function_names, title=title,
                                       subtitle=f"average of {number_of_iterations} iterations")


def vs_multi(env, number_of_iterations, functions, function_names, title):
    fitness_curves_list = list()
    for j in range(len(functions)):
        fitness_curves = list()
        start_time = time.perf_counter()
        for i in range(0, number_of_iterations):
            _, fitness_curve = functions[j](env)
            fitness_curves.append(fitness_curve)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"The execution time of {function_names[j]} is: {execution_time / number_of_iterations}")
        fitness_curves_list.append(fitness_curves)

    sketch.show_vs_multi_fitness_curve(fitness_curves_list, function_names, title=title,
                                       subtitle=f"average of {number_of_iterations} iterations")


def execute_monte_carlo(env):
    start_time = time.perf_counter()
    v, fitness_curve = monte_carlo_algorithm.monte_carlo(env, episodes=500)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}")
    print(f"V: \n{v}")
    print(f"pi: \n{monte_carlo_algorithm.get_pi_from_v(env, v)}")
    sketch.show_fitness_curve(fitness_curve, subtitle="Monte Carlo")


def avg_monte_carlo(env, number_of_iterations):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = monte_carlo_algorithm.monte_carlo(env, episodes=500)
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Monte Carlo",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_td_0(env):
    v, fitness_curve = value_algorithm.td_0(env, epsilon=0.4)
    print(v)
    print(f"pi: \n{value_algorithm.get_pi_from_v(env, v)}")
    sketch.show_fitness_curve(fitness_curve, subtitle="TD(0)")


def avg_td_0(env, number_of_iterations):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = value_algorithm.td_0(env, epsilon=0.4)
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with TD(0)",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_td_n(env, n):
    v, fitness_curve = value_algorithm.td_n(env, n)
    print(v)
    sketch.show_fitness_curve(fitness_curve, subtitle=f"TD({n})")


def avg_td_n(env, n, number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve = value_algorithm.td_n(env, n)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title=f"Fitness Curve with TD({n})",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_q_learning(env):
    quality_table, fitness_curve = quality_algorithm.q_learning(env)
    for i in range(len(quality_table)):
        print(f"State {env.int_state_to_tuple(i)}: {quality_table[i]}")
    print(f"pi: \n{quality_algorithm.get_pi_from_q(env, quality_table)}")
    sketch.show_fitness_curve(fitness_curve, subtitle="Q-Learning")


def avg_q_learning(env, number_of_iterations):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = quality_algorithm.q_learning(env)
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Q-Learning",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_sarsa(env):
    quality_table, fitness_curve = quality_algorithm.sarsa(env)
    for i in range(len(quality_table)):
        print(f"State {env.int_state_to_tuple(i)}: {quality_table[i]}")
    print(f"pi: \n{quality_algorithm.get_pi_from_q(env, quality_table)}")
    sketch.show_fitness_curve(fitness_curve, subtitle="SARSA")


def avg_sarsa(env, number_of_iterations):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = quality_algorithm.sarsa(env)
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with SARSA",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_function(function, env=None, n=None, episodes=None, gamma=None, epsilon=None, alpha=None,
                     epsilon_decay=None, updates=False):
    nn = n is None
    epin = episodes is None
    gn = gamma is None
    en = epsilon is None
    an = alpha is None
    edn = epsilon_decay is None

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
                            return function(env, gamma=gamma, alpha=alpha, epsilon_decay=epsilon_decay, updates=updates)
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
                            return function(env, gamma=gamma, epsilon=epsilon, alpha=alpha, epsilon_decay=epsilon_decay,
                                            updates=updates)
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
                                            epsilon_decay=epsilon_decay,
                                            updates=updates)
        else:
            if gn:
                if en:
                    if an:
                        if edn:
                            return function(env, n=n, episodes=episodes, updates=updates)
                        else:
                            return function(env, n=n, episodes=episodes, epsilon_decay=epsilon_decay, updates=updates)
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
                            return function(env, n=n, episodes=episodes, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                            updates=updates)
                    else:
                        if edn:
                            return function(env, n=n, episodes=episodes, epsilon=epsilon, alpha=alpha, updates=updates)
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
                            return function(env, n=n, episodes=episodes, gamma=gamma, epsilon=epsilon, updates=updates)
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


if __name__ == '__main__':
    main()
