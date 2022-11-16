import monte_carlo_algorithm
import sketch
import quality_algorithm
import value_algorithm
import environemnt


def main():
    labyrinth = environemnt.Labyrinth()
    tictactoe = environemnt.TicTacToe()

    execute_q_learning(tictactoe)

    # avg_monte_carlo(100)
    # avg_td_0(100)
    # avg_td_n(10, 100)
    # avg_q_learning(100)
    # avg_sarsa(100)
    # execute_td_0()


def execute_monte_carlo(env):
    v, fitness_curve, pi = monte_carlo_algorithm.monte_carlo(env)
    print(pi)
    print(v)
    sketch.show_fitness_curve(fitness_curve, subtitle="Monte Carlo")


def avg_monte_carlo(env, number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = monte_carlo_algorithm.monte_carlo(env)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Monte Carlo",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_td_0(env):
    v, fitness_curve, pi = value_algorithm.td_0(env, updates=True)
    print(pi)
    print(v)
    sketch.show_fitness_curve(fitness_curve, subtitle="TD(0)")


def avg_td_0(env, number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = value_algorithm.td_0(env, epsilon=0.4)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with TD(0)",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_td_n(env, n):
    v, fitness_curve, pi = value_algorithm.td_n(env, n)
    print(pi)
    print(v)
    sketch.show_fitness_curve(fitness_curve, subtitle=f"TD({n})")


def avg_td_n(emv, n, number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = value_algorithm.td_n(env, n, epsilon=0.4)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title=f"Fitness Curve with TD({n})",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_q_learning(env):
    qualityTable, fitness_curve, pi = quality_algorithm.q_learning(env, epsilon=0.4)
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i % 6},{i // 6}): {q}")
        i += 1
    sketch.show_fitness_curve(fitness_curve, subtitle="Q-Learning")


def avg_q_learning(env, number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = quality_algorithm.q_learning(env, epsilon=0.4)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Q-Learning",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_sarsa(env):
    qualityTable, fitness_curve, pi = quality_algorithm.sarsa(env)
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i // 6},{i % 6}): {q}")
        i += 1
    sketch.show_fitness_curve(fitness_curve, subtitle="SARSA")


def avg_sarsa(env, number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = quality_algorithm.sarsa(env)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with SARSA",
                                  subtitle=f"average of {number_of_iterations} iterations")


if __name__ == '__main__':
    main()
