import numpy as np
import numpy.random as random
import util


def getPi(env, qTable):
    pi = {}
    state = (0, 0)

    while state[0] < env.height:
        while state[1] < env.width:

            if env.getR(state, (0, 0)) == -np.inf:
                pi[state] = "-"
                state = (state[0], state[1] + 1)
                continue

            possibleActions = []
            possibleQ = []
            for action in env.actions:
                if env.getR(state, action) != -np.inf:
                    possibleActions.append(action)
                    possibleQ.append(qTable[env.stateToInt(state), env.actionToInt(action)])
            maxQuality = np.max(possibleQ)
            # intPosActions = [env.actionToInt(pa) for pa in possibleActions]
            bestActions = [possibleActions[i] for i in range(0, len(possibleActions)) if possibleQ[i] == maxQuality]
            pi[state] = [env.actionToStr(ba) for ba in bestActions]

            state = (state[0], state[1] + 1)
        state = (state[0] + 1, 0)

    return pi


def genericQualityAlgorithm(env, function, episodes, gamma, epsilon, alpha, epsilondecay, updates):
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

            reward = env.getR(state, action)
            theReturn += reward

            # Update Q value
            qTable[intState, intAction] = function(env, reward, qTable[intState, intAction],
                                                   qTable[env.stateToInt(nextState), :], nextState,
                                                   gamma, alpha, epsilon)

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


def qLearningFunction(env, reward, qSA, qnS, nS, gamma, alpha, epsilon):
    TDTargetEstimate = reward + gamma * qnS.max()
    TDError = TDTargetEstimate - qSA
    return qSA + alpha * TDError


def qLearning(env, episodes=1000, gamma=0.8, epsilon=1, alpha=0.1, epsilondecay=0.99, updates=False):
    function = qLearningFunction
    return genericQualityAlgorithm(env, function, episodes, gamma, epsilon, alpha, epsilondecay, updates)


def sarsaFunction(env, reward, qSA, qnS, nS, gamma, alpha, epsilon):
    # choose a possible action
    # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
    possibleActions = []
    possibleQ = []
    for action in env.actions:
        if env.getR(nS, action) != -np.inf:
            possibleActions.append(action)
            possibleQ.append(qnS[env.actionToInt(action)])

    # Step next state, here we use epsilon-greedy algorithm.
    if random.random() < epsilon:
        # choose random action
        nextAction = possibleActions[random.randint(0, len(possibleActions))]
    else:
        # greedy
        nextAction = possibleActions[util.argmax(possibleQ)]

    TDTargetEstimate = reward + gamma * qnS[env.actionToInt(nextAction)]
    TDError = TDTargetEstimate - qSA
    return qSA + alpha * TDError


def sarsa(env, episodes=1000, gamma=0.8, epsilon=1, alpha=0.1, epsilondecay=0.99, updates=False):
    function = sarsaFunction
    return genericQualityAlgorithm(env, function, episodes, gamma, epsilon, alpha, epsilondecay, updates)
