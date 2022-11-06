import numpy as np
import numpy.random as random
import util


def getPi(env, qTable):
    pi = []
    state = env.startState

    while state != env.finalState:
        possibleActions = []
        possibleQ = []
        for action in env.actions:
            if env.getR(state, action) != -np.inf:
                possibleActions.append(action)
                possibleQ.append(qTable[env.stateToInt(state), env.actionToInt(action)])

        action = possibleActions[util.argmax(possibleQ)]
        pi.append(env.actionToStr(action))
        state = env.getNextState(state, action)
        if len(pi) > 1000:
            pi.append("Result is to long")
            break

    return pi


def qLearning(env, episodes=1000, gamma=0.99, epsilon=0.9, alpha=0.05, epsilondecay=0.99, updates=False):
    fitnessCurve = []

    # qTable
    qTable = np.full([env.height * env.width, len(env.actions)], 0, dtype=float)

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.startState

        theReturn = 0

        # if not final state
        while state != env.finalState:
            intState = env.stateToInt(state)

            # choose a possible action
            # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
            possibleActions = []
            possibleQ = []
            for action in env.actions:
                if env.getR(state, action) != -np.inf:
                    possibleActions.append(action)
                    possibleQ.append(qTable[intState, env.actionToInt(action)])

            # Step next state, here we use epsilon-greedy algorithm.
            if random.random() < epsilon:
                # choose random action
                action = possibleActions[random.randint(0, len(possibleActions))]
            else:
                # greedy
                action = possibleActions[util.argmax(possibleQ)]

            intAction = env.actionToInt(action)
            nextState = env.getNextState(state, action)

            # Update Q value
            currentReward = env.getR(state, action)
            TDTargetEstimate = currentReward + gamma * qTable[env.stateToInt(nextState)].max()
            TDError = TDTargetEstimate - qTable[intState, intAction]
            qTable[intState, intAction] = qTable[intState, intAction] + alpha * TDError

            theReturn += currentReward

            # Go to the next state
            state = nextState

        fitnessCurve.append(theReturn)

        epsilon *= epsilondecay
        # Display training progress
        if episode % 100 == 0 and updates:
            print("Training episode: %d" % episode)
            print(qTable)
            # print("     -       -       -")
            print(f"Current epsilon: {epsilon}")
            print(f"the return: {theReturn}")
            print("------------------------------------------------")

    return qTable, fitnessCurve, getPi(env, qTable)
