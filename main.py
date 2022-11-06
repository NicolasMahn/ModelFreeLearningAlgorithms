import sketch
import quality_function
import value_function
import environemnt


def main():
    execute_td_n(0)


def execute_monte_carlo():
    v, fitness_curve, pi = value_function.monte_carlo(environemnt.Labyrinth())
    print(pi)
    print(v)
    sketch.showList(fitness_curve, subtitle="Monte Carlo")


def execute_td_0():
    v, fitness_curve, pi = value_function.td_0(environemnt.Labyrinth())
    print(pi)
    print(v)
    sketch.showList(fitness_curve, subtitle="TD(0)")


def execute_td_n(n):
    v, fitness_curve, pi = value_function.td_n(environemnt.Labyrinth(), n)
    print(pi)
    print(v)
    sketch.showList(fitness_curve, subtitle=f"TD({n})")


def execute_q_learning():
    qualityTable, fitness_curve, pi = quality_function.q_learning(environemnt.Labyrinth())
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i // 6},{i % 6}): {q}")
        i += 1
    sketch.showList(fitness_curve, subtitle="Q-Learning")


def execute_sarsa():
    qualityTable, fitness_curve, pi = quality_function.sarsa(environemnt.Labyrinth())
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i // 6},{i % 6}): {q}")
        i += 1
    sketch.showList(fitness_curve, subtitle="SARSA")


if __name__ == '__main__':
    main()
