import gym_ttt.gym_controller
import algorithms as alg
import time
import environments
from sketch.sketch import show_fitness_curve, show_vs2_fitness_curve, execute_function, show_vs_multi_fitness_curve, \
    show_avg_fitness_curve, show_avg_tictactoe


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

    show_vs2_fitness_curve(fitness_curves_main, main_name, fitness_curves_sub, sub_name,
                           title=f"Fitness Curve comparing {main_name} with {sub_name}",
                           subtitle=f"average of {number_of_iterations} iterations")


def vs_multi_with_par(env, number_of_iterations, functions, function_names, title="title",
                      n_list=None, episodes_list=None, gamma_list=None, epsilon_list=None,
                      alpha_list=None, epsilon_decay_list=None, prev_state_list=None):
    if n_list is None:
        n_list = [None] * len(functions)
    elif isinstance(n_list, int):
        n_list = [n_list] * len(functions)

    if episodes_list is None:
        episodes_list = [None] * len(functions)
    elif isinstance(episodes_list, int):
        episodes_list = [episodes_list] * len(functions)

    if gamma_list is None:
        gamma_list = [None] * len(functions)
    elif isinstance(gamma_list, int) or isinstance(gamma_list, float):
        gamma_list = [gamma_list] * len(functions)

    if epsilon_list is None:
        epsilon_list = [None] * len(functions)
    elif isinstance(epsilon_list, int) or isinstance(epsilon_list, float):
        epsilon_list = [epsilon_list] * len(functions)

    if alpha_list is None:
        alpha_list = [None] * len(functions)
    elif isinstance(alpha_list, int) or isinstance(alpha_list, float):
        alpha_list = [alpha_list] * len(functions)

    if epsilon_decay_list is None:
        epsilon_decay_list = [None] * len(functions)
    elif isinstance(epsilon_decay_list, int) or isinstance(epsilon_decay_list, float):
        epsilon_decay_list = [epsilon_decay_list] * len(functions)

    if prev_state_list is None:
        prev_state_list = [None] * len(functions)
    elif isinstance(prev_state_list, bool):
        prev_state_list = [prev_state_list] * len(functions)

    fitness_curves_list = list()
    for j in range(len(functions)):
        fitness_curves = list()
        start_time = time.perf_counter()
        for i in range(0, number_of_iterations):
            _, fitness_curve = execute_function(function=functions[j], env=env, n=n_list[j], episodes=episodes_list[j],
                                                gamma=gamma_list[j], epsilon=epsilon_list[j], alpha=alpha_list[j],
                                                epsilon_decay=epsilon_decay_list[j], prev_state=prev_state_list[j])
            fitness_curves.append(fitness_curve)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"The execution time of {function_names[j]} is: {execution_time / number_of_iterations}")
        fitness_curves_list.append(fitness_curves)

    show_vs_multi_fitness_curve(fitness_curves_list, function_names, title=title,
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

    show_vs_multi_fitness_curve(fitness_curves_list, function_names, title=title,
                                subtitle=f"average of {number_of_iterations} iterations")


def execute_monte_carlo(env):
    start_time = time.perf_counter()
    v, fitness_curve = alg.monte_carlo(env)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}")
    print(f"V: \n{v}")
    print(f"pi: \n{alg.get_pi_from_v(env, v)}")
    show_fitness_curve(fitness_curve, subtitle="Monte Carlo")


def avg_monte_carlo(env, number_of_iterations):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = alg.monte_carlo(env)
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Monte Carlo",
                           subtitle=f"average of {number_of_iterations} iterations")


def execute_td_0(env):
    start_time = time.perf_counter()
    v, fitness_curve = alg.td_0(env)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}")
    print(f"V: \n{v}")
    print(f"pi: \n{alg.get_pi_from_v(env, v)}")
    show_fitness_curve(fitness_curve, subtitle="TD(0)")


def avg_td_0(env, number_of_iterations):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = alg.td_0(env)
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    show_avg_fitness_curve(fitness_curves, title="Fitness Curve with TD(0)",
                           subtitle=f"average of {number_of_iterations} iterations")


def execute_td_n(env, n):
    start_time = time.perf_counter()
    v, fitness_curve = alg.td_n(env, n)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}")
    print(f"V: \n{v}")
    print(f"pi: \n{alg.get_pi_from_v(env, v)}")
    show_fitness_curve(fitness_curve, subtitle=f"TD({n})")


def avg_td_n(env, n, number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve = alg.td_n(env, n)
        fitness_curves.append(fitness_curve)
        print(i)

    show_avg_fitness_curve(fitness_curves, title=f"Fitness Curve with TD({n})",
                           subtitle=f"average of {number_of_iterations} iterations")


def execute_q_learning(env):
    start_time = time.perf_counter()
    quality_table, fitness_curve = alg.q_learning(env)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}")
    print(f"V: \n{alg.get_v_from_q(env, quality_table)}")
    print(f"pi: \n{alg.get_pi_from_q(env, quality_table)}")
    show_fitness_curve(fitness_curve, subtitle="Q-Learning")


def avg_q_learning(env, number_of_iterations, episodes=500):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = alg.q_learning(env, episodes)
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Q-Learning",
                           subtitle=f"average of {number_of_iterations} iterations")


def avg_q_learning_tictactoe(player=1, number_of_iterations=1000, episodes=1000, smart=False, title="TicTacToe",
                             epsilon=0, epsilon_decay=1, update=True):
    if smart:
        env = environments.TictactoeVS()
    else:
        env = environments.TicTacToe(player)

    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        if smart:
            return_tuple = alg.q_learning_vs(env, episodes, epsilon=epsilon, epsilon_decay=epsilon_decay)
        else:
            return_tuple = alg.q_learning(env, episodes, epsilon=epsilon, epsilon_decay=epsilon_decay)
        fitness_curves.append(return_tuple[-1])
        if i+1 % 100 == 0 and update:
            print(f"Currently calculating iteration: {i+1}")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    if smart:
        show_avg_tictactoe(fitness_curves, title=title,
                           subtitle=f"average of {number_of_iterations} iterations")
    else:
        show_avg_tictactoe(fitness_curves, title=title,
                           subtitle=f"average of {number_of_iterations} iterations",
                           player_flag=100, opponent_flag=-100)


def execute_sarsa(env):
    quality_table, fitness_curve = alg.sarsa(env)
    print(f"pi: \n{alg.get_pi_from_q(env, quality_table)}")
    print(f"v: \n{alg.get_v_from_q(env, quality_table)}")
    print(f"Q-Table for LaTeX:")
    get_latex_table_from_q_table(env, quality_table)
    show_fitness_curve(fitness_curve, subtitle="SARSA")


def avg_sarsa(env, number_of_iterations):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = alg.sarsa(env)
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    show_avg_fitness_curve(fitness_curves, title="Fitness Curve with SARSA",
                           subtitle=f"average of {number_of_iterations} iterations")


def get_latex_table_from_q_table(env, q_table):
    s = r'''\begin{minipage}{1\textwidth}
                \centering
                \begin{tabular}{|cc|c|c|c|c|c|c|c|c|}
                    \hline
                    & & \multicolumn{7}{c}{\textbf{actions}} & \\
                    \cline{3-10}
                    & & \textbf{''' + env.int_action_to_str(0) + \
        r'} & \textbf{' + env.int_action_to_str(1) + \
        r'} & \textbf{' + env.int_action_to_str(2) + \
        r'} & \textbf{' + env.int_action_to_str(3) + \
        r'} & \textbf{' + env.int_action_to_str(4) + \
        r'} & \textbf{' + env.int_action_to_str(5) + \
        r'} & \textbf{' + env.int_action_to_str(6) + \
        r'} & \textbf{' + env.int_action_to_str(7) + r'} \\'
    s += r'''\hline
             \multirow{25}{*}{\begin{sideways}\textbf{states}\end{sideways}}'''

    for state in range(len(q_table)):
        s += r'&\multicolumn{1}{|c|}{\textbf{'
        s += str(env.int_state_to_tuple(state))
        s += r'}}'
        for quality in q_table[state]:
            s += '& '
            s += str(quality)
        s += r'\\'
        if state < len(q_table) - 1:
            s += r'\cline{2-10}'
            s += '\n'
    s += '''\hline
    \end{tabular}
    \captionof{table}{Q-Table}
\end{minipage}  '''
    print(s)


def execute_gym():
    quality_table, fitness_curve = gym_ttt.gym_controller.train()
    show_fitness_curve(fitness_curve, subtitle="Q-Learning with gym")


def avg_gym(env, number_of_iterations, episodes=500):
    fitness_curves = list()
    start_time = time.perf_counter()
    for i in range(0, number_of_iterations):
        _, fitness_curve = gym_ttt.gym_controller.train()
        fitness_curves.append(fitness_curve)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time / number_of_iterations}")

    show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Q-Learning",
                           subtitle=f"average of {number_of_iterations} iterations")
