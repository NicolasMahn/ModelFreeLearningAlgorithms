import math

import numpy as np
import numpy.random as random
import environemnt
import util


def getPi(env, v):
    pi = {}
    state = (0, 0)

    while state[0] < env.height:
        while state[1] < env.width:
            possibleActions = []
            for action in env.actions:
                if env.getR(state, action) != -np.inf:
                    possibleActions.append(action)
            maxV = np.max([v[env.getNextState(state, a)] for a in possibleActions])
            bestActions = [pa for pa in possibleActions if v[env.getNextState(state, pa)] == maxV]
            pi[state] = [env.actionToStr(ba) for ba in bestActions]

            state = (state[0], state[1] + 1)
        state = (state[0] + 1, 0)

    return pi


def genericValueAlgorithm(env, n, function, episodes, gamma, epsilon, alpha, epsilondecay, updates):
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
            v[state] = function(env, v, n, reward, state, nextState, theReturn, episode, gamma, alpha)

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


def tdNFunction(env, v, n, reward, state, nextState, theReturn, episode, gamma, alpha):
    returnN = reward
    vNextState = v[nextState]
    for cn in range(1,n):
        possibleActions = [action for action in env.actions if env.getR(nextState, action) != -np.inf]
        action = possibleActions[util.argmax([v[env.getNextState(nextState, a)] for a in possibleActions])]
        returnN += math.pow(gamma, cn)*env.getR(nextState, action)
        nextState = env.getNextState(nextState, action)
        vNextState = v[nextState]

    TDTargetEstimate = reward + gamma * vNextState
    TDError = TDTargetEstimate - v[state]
    return v[state] + alpha * TDError


def tdN(env, n, episodes=1000, gamma=0.5, epsilon=0.9, alpha=0.05, epsilondecay=0.99, updates=False):
    function = tdNFunction
    return genericValueAlgorithm(env, n, function, episodes, gamma, epsilon, alpha, epsilondecay, updates)


def td0Function(env, v, n, reward, state, nextState, theReturn, episode, gamma, alpha):
    TDTargetEstimate = reward + gamma * v[nextState]
    TDError = TDTargetEstimate - v[state]
    return v[state] + alpha * TDError


def td0(env, episodes=1000, gamma=0.5, epsilon=0.9, alpha=0.05, epsilondecay=0.99, updates=False):
    function = td0Function
    return genericValueAlgorithm(env, 0, function, episodes, gamma, epsilon, alpha, epsilondecay, updates)


def monteCarloFunction(env, v, n, reward, state, nextState, theReturn, episode, gamma, alpha):
    return v[state] + (1 / (episode + 1)) * (theReturn + v[state])


def monteCarlo(env, episodes=1000, epsilon=0.9, epsilondecay=0.99, updates=False):
    function = monteCarloFunction
    return genericValueAlgorithm(env, function, episodes, None, epsilon, None, epsilondecay, updates)
