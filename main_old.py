import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

"""
Code was inspired by:
https://medium.com/ai%C2%B3-theory-practice-business/reinforcement-learning-part-6-td-%CE%BB-q-learning-99cdfdf4e76a
"""


# Labyrinth width
lw = 4
# Labyrinth height
lh = 6

# Actions
a = [(-1, -1),  #NW
     (-1, 0),   #N
     (-1, 1),   #NE
     (0, -1),   #W
     (0, 1),    #E
     (1, -1),   #SW
     (1, 0),    #S
     (1, 1)]    #SE

# Q-Table
#q = np.matrix(np.zeros([lh*lw, len(a)]))
q = np.full([lh*lw, len(a)], 0) #, dtype=float)

# Reward
r = np.matrix([[-1,  -1,      -1,      -1],
               [-1,  -np.inf,  60,     -1],
               [-1,  -np.inf, -20,     -np.inf],
               [-1,  -1,      -1,      -1],
               [-70, -10,     -np.inf, -1],
               [-1,  -1,      -1,      -1]])


# since rewards are actually given for state action pairs
def getR(state, action):
    nextState = getNextState(state, action)
    if nextState[0] >= lh or nextState[0] < 0 or nextState[1] >= lw or nextState[1] < 0:
        return -np.inf
    else:
        return r.item(nextState)

def getNextState(state, action):
    return (state[0]+action[0], state[1]+action[1])

# since state is a tuple in this example but not in the QTable
def stateToInt(state):
    return state[0]*lw+state[1]

# since action is a tuple in this example but not in the QTable
def actionToInt(action):
    return a.index(action)


def main():
    gamma = 0.5
    epsilon = 0.4
    alpha = 0.05

    fitnessCurve = []

    # the main training loop
    for episode in range(1001):

        # initial state
        state = (5, 0)

        theWay = []
        theReturn = 0

        # if not final state
        while state != (1, 2):

            # choose a possible action
            # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
            possibleActions = []
            possibleQ = []
            for action in a:
                if getR(state, action) != -np.inf:
                    possibleActions.append(action)
                    possibleQ.append(q[stateToInt(state), actionToInt(action)])

            # Step next state, here we use epsilon-greedy algorithm.
            if rand.random() < epsilon:
                # choose random action
                action = possibleActions[rand.randint(0, len(possibleActions))]
            else:
                # greedy
                maxq = np.max(possibleQ)

                bestQs = []
                l = 0
                for pq in possibleQ:
                    if pq == maxq:
                        bestQs.append(l)
                    l += 1

                action = possibleActions[bestQs[rand.randint(0, len(bestQs))]]

            # Update Q value
            currentReward = getR(state, action)
            TDTargetEstimate = currentReward + gamma * (q[stateToInt(getNextState(state, action))].max())
            TDError = TDTargetEstimate -q[stateToInt(state), actionToInt(action)]
            #q[stateToInt(state), actionToInt(action)] = currentReward + gamma * q[action].max()
            q[stateToInt(state), actionToInt(action)] = q[stateToInt(state), actionToInt(action)] + alpha*TDError

            theWay.append(state)
            theReturn += currentReward

            # Go to the next state
            state = getNextState(state, action)

        fitnessCurve.append(theReturn)

        epsilon *= 0.99
        # Display training progress
        if episode % 10 == 0:
            print("Training episode: %d" % episode)
            # print(q)
            # print("     -       -       -")
            print(epsilon)
            print(f"length of the Way: {len(theWay)}")
            print(f"the return: {theReturn}")
            print("------------------------------------------------")

    print("-----------------DONE---------------------------")
    print(q)

    i = 0
    j = 0
    actions = ["NW","N","NE","W","E","SW","S","SE"]

    for s in q:
        possibleA = []
        possibleQu = []
        for k in range(0,len(s)):
            if getR((j,i), a[k]) != -np.inf:
                possibleA.append(k)
                possibleQu.append(q[stateToInt((j,i)), k])
        qual = -np.inf
        bestA = ""
        for qu in possibleQu:
            if qu > qual:
                qual = qu

        lis = []
        l = 0
        for pq in possibleQu:
            if pq == qual:
               lis.append(l)
            l +=1

        for n in lis:
            bestA += f" {actions[possibleA[n]]}"

        print(f"State ({j},{i}) best action:{bestA}")
        i += 1
        if i == lw:
            i = 0
            j += 1

    fig = plt.figure()
    plt.title("Fitness Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.plot([fitnessCurve[i] for i in range(0, len(fitnessCurve))], color="black")
    plt.show()


if __name__ == '__main__':
    main()

