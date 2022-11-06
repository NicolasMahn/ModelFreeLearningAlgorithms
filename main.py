import MonteCarlo
import sketch
import qualityFunction
import valueFunction
import MonteCarlo
import environemnt


def main():
    executeTD0()


def executeMonteCarlo():
    v, fitnessCurve, pi = MonteCarlo.monteCarlo(environemnt.Labyrinth(), updates=True)
    print(pi)
    print(v)
    sketch.showList(fitnessCurve, subtitle="Monte Carlo")


def executeTD0():
    v, fitnessCurve, pi = valueFunction.td0(environemnt.Labyrinth())
    print(pi)
    print(v)
    sketch.showList(fitnessCurve, subtitle="TD(0)")


def executeQ():
    qualityTable, fitnessCurve, pi = qualityFunction.qLearning(environemnt.Labyrinth())
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i // 9},{i % 9}): {q}")
        i += 1
    sketch.showList(fitnessCurve, subtitle="Q-Learning")


if __name__ == '__main__':
    main()
