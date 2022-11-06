import numpy as np
import numpy.random as random
import environemnt
import util


def getPi(env, v):
    pi = []
    state = env.startState

    while state != env.finalState:
        possibleActions = []
        for action in env.actions:
            if env.getR(state, action) != -np.inf:
                possibleActions.append(action)

        action = possibleActions[util.argmax([v[env.getNextState(state, a)] for a in possibleActions])]
        pi.append(env.actionToStr(action))
        state = env.getNextState(state, action)
        if len(pi) > 1000:
            pi.append("Result is to long")
            break

    return pi


def genericAlgorithm(env, function, episodes, gamma, epsilon, alpha, epsilondecay, updates):
    fitnessCurve = []

    # Value States
    v = np.full([env.height, env.width], 0, dtype=float)

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
            for action in env.actions:
                if env.getR(state, action) != -np.inf:
                    possibleActions.append(action)

            # Step next state, here we use epsilon-greedy algorithm.
            if random.random() < epsilon:
                # choose random action
                action = possibleActions[random.randint(0, len(possibleActions))]
            else:
                # greedy
                action = possibleActions[util.argmax([v[env.getNextState(state, a)] for a in possibleActions])]

            nextState = env.getNextState(state, action)

            reward = env.getR(state, action)
            theReturn += reward

            # Update V value
            v[state] = function(reward, v[state], v[nextState], gamma, alpha)

            # Go to the next state
            state = nextState

        fitnessCurve.append(theReturn)

        epsilon *= epsilondecay
        # Display training progress
        if episode % 100 == 0 and updates:
            print("Training episode: %d" % episode)
            print(v)
            # print("     -       -       -")
            print(f"Current epsilon: {epsilon}")
            print(f"the return: {theReturn}")
            print("------------------------------------------------")

    return v, fitnessCurve, getPi(env, v)


def td0Function(reward, vS, vnS, gamma, alpha):
    TDTargetEstimate = reward + gamma * vnS
    TDError = TDTargetEstimate - vS
    return vS + alpha * TDError


def td0(env, episodes=1000, gamma=0.5, epsilon=0.9, alpha=0.05, epsilondecay=0.99, updates=False):
    function = td0Function
    return genericAlgorithm(env, function, episodes, gamma, epsilon, alpha, epsilondecay, updates)



def monteCarlo(env, episodes=1000, epsilon=0.9, epsilondecay=0.99, updates=False):
    fitnessCurve = []

    # Value States
    v = np.full([env.height, env.width], 0, dtype=float)

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
            for action in env.actions:
                if env.getR(state, action) != -np.inf:
                    possibleActions.append(action)

            # Step next state, here we use epsilon-greedy algorithm.
            if random.random() < epsilon:
                # choose random action
                action = possibleActions[random.randint(0, len(possibleActions))]
            else:
                # greedy
                action = possibleActions[util.argmax([v[env.getNextState(state, a)] for a in possibleActions])]

            nextState = env.getNextState(state, action)

            currentReward = env.getR(state, action)
            theReturn += currentReward

            # Update V value
            v[state] = v[state] + (1 / (episode + 1)) * (theReturn + v[state])

            # Go to the next state
            state = nextState

        fitnessCurve.append(theReturn)

        epsilon *= epsilondecay
        # Display training progress
        if episode % 100 == 0 and updates:
            print("Training episode: %d" % episode)
            print(v)
            # print("     -       -       -")
            print(f"Current epsilon: {epsilon}")
            print(f"the return: {theReturn}")
            print("------------------------------------------------")

    return v, fitnessCurve, getPi(env, v)
