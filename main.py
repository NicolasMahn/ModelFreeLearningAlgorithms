import monte_carlo_algorithm
import sketch
import quality_algorithm
import value_algorithm
import environemnt


def main():
    # avg_monte_carlo(100)
    # avg_td_0(100)
    # avg_td_n(10, 100)
    avg_q_learning(100)
    # avg_sarsa(100)
    # execute_td_0()


def execute_monte_carlo():
    v, fitness_curve, pi = monte_carlo_algorithm.monte_carlo(environemnt.Labyrinth())
    print(pi)
    print(v)
    sketch.show_fitness_curve(fitness_curve, subtitle="Monte Carlo")


def avg_monte_carlo(number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = monte_carlo_algorithm.monte_carlo(environemnt.Labyrinth(), epsilon=0.4)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Monte Carlo",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_td_0():
    v, fitness_curve, pi = value_algorithm.td_0(environemnt.Labyrinth(), updates=True)
    print(pi)
    print(v)
    sketch.show_fitness_curve(fitness_curve, subtitle="TD(0)")


def avg_td_0(number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = value_algorithm.td_0(environemnt.Labyrinth(), epsilon=0.4)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with TD(0)",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_td_n(n):
    v, fitness_curve, pi = value_algorithm.td_n(environemnt.Labyrinth(), n)
    print(pi)
    print(v)
    sketch.show_fitness_curve(fitness_curve, subtitle=f"TD({n})")


def avg_td_n(n, number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = value_algorithm.td_n(environemnt.Labyrinth(), n, epsilon=0.4)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title=f"Fitness Curve with TD({n})",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_q_learning():
    qualityTable, fitness_curve, pi = quality_algorithm.q_learning(environemnt.Labyrinth(), epsilon=0.4)
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i % 6},{i // 6}): {q}")
        i += 1
    sketch.show_fitness_curve(fitness_curve, subtitle="Q-Learning")


def avg_q_learning(number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = quality_algorithm.q_learning(environemnt.Labyrinth(), epsilon=0.4)
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with Q-Learning",
                                  subtitle=f"average of {number_of_iterations} iterations")


def execute_sarsa():
    qualityTable, fitness_curve, pi = quality_algorithm.sarsa(environemnt.Labyrinth())
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i // 6},{i % 6}): {q}")
        i += 1
    sketch.show_fitness_curve(fitness_curve, subtitle="SARSA")


def avg_sarsa(number_of_iterations):
    fitness_curves = list()
    for i in range(0, number_of_iterations):
        _, fitness_curve, _ = quality_algorithm.sarsa(environemnt.Labyrinth())
        fitness_curves.append(fitness_curve)
        print(i)

    sketch.show_avg_fitness_curve(fitness_curves, title="Fitness Curve with SARSA",
                                  subtitle=f"average of {number_of_iterations} iterations")


if __name__ == '__main__':
    main()
