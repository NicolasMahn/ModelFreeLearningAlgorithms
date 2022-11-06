import sketch
import qualityFunction
import valueFunction
import environemnt


def main():
    executeTDN(0)


def executeMonteCarlo():
    v, fitnessCurve, pi = valueFunction.monteCarlo(environemnt.Labyrinth())
    print(pi)
    print(v)
    sketch.showList(fitnessCurve, subtitle="Monte Carlo")


def executeTD0():
    v, fitnessCurve, pi = valueFunction.td0(environemnt.Labyrinth())
    print(pi)
    print(v)
    sketch.showList(fitnessCurve, subtitle="TD(0)")


def executeTDN(n):
    v, fitnessCurve, pi = valueFunction.tdN(environemnt.Labyrinth(),n)
    print(pi)
    print(v)
    sketch.showList(fitnessCurve, subtitle=f"TD({n})")

def executeQ():
    qualityTable, fitnessCurve, pi = qualityFunction.qLearning(environemnt.Labyrinth())
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i // 6},{i % 6}): {q}")
        i += 1
    sketch.showList(fitnessCurve, subtitle="Q-Learning")


def executeSARSA():
    qualityTable, fitnessCurve, pi = qualityFunction.sarsa(environemnt.Labyrinth())
    print(pi)
    i = 0
    for q in qualityTable:
        print(f"State ({i // 6},{i % 6}): {q}")
        i += 1
    sketch.showList(fitnessCurve, subtitle="SARSA")

if __name__ == '__main__':
    main()
