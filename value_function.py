import math

import numpy as np
import numpy.random as random
import environemnt
import util


def get_pi(env, v):
    pi = {}
    state = (0, 0)

    while state[0] < env.height:
        while state[1] < env.width:

            if env.get_r(state, (0, 0)) == -np.inf:
                pi[state] = "-"
                state = (state[0], state[1] + 1)
                continue

            possible_actions = get_possible_actions(env, state)
            maxV = np.max([v[env.get_next_state(state, a)] for a in possible_actions])
            best_actions = [pa for pa in possible_actions if v[env.get_next_state(state, pa)] == maxV]
            pi[state] = [env.action_to_str(ba) for ba in best_actions]

            state = (state[0], state[1] + 1)
        state = (state[0] + 1, 0)

    return pi


def get_possible_actions(env, state):
    return [action for action in env.actions if env.get_r(state, action) != -np.inf]


def generic_value_algorithm(env, n, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates):
    fitness_curve = []

    # Value States
    v = np.full([env.height, env.width], 0, dtype=float)

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.start_state

        return_ = 0

        # if not final state
        while state != env.final_state:

            # choose a possible action
            # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
            possible_actions = get_possible_actions(env, state)

            # Step next state, here we use epsilon-greedy algorithm.
            if random.random() < epsilon:
                # choose random action
                action = possible_actions[random.randint(0, len(possible_actions))]
            else:
                # greedy
                action = possible_actions[util.argmax([v[env.get_next_state(state, a)] for a in possible_actions])]

            next_state = env.get_next_state(state, action)

            reward = env.get_r(state, action)
            return_ += reward

            # Update V value
            v[state] = function(env, v, n, reward, state, next_state, return_, episode, gamma, alpha)

            # Go to the next state
            state = next_state

        fitness_curve.append(return_)
        epsilon *= epsilon_decay

        # Display training progress
        if episode % 100 == 0 and updates:
            print("Training episode: %d" % episode)
            print(v)
            # print("     -       -       -")
            print(f"Current epsilon: {epsilon}")
            print(f"the return: {return_}")
            print("------------------------------------------------")

    return v, fitness_curve, get_pi(env, v)


def td_n_function(env, v, n, reward, state, next_state, return_, episode, gamma, alpha):
    returnN = reward
    v_next_state = v[next_state]
    for cn in range(1, n):
        possible_actions = get_possible_actions(env, next_state)
        action = possible_actions[util.argmax([v[env.get_next_state(next_state, a)] for a in possible_actions])]
        returnN += math.pow(gamma, cn)*env.get_r(next_state, action)
        next_state = env.get_next_state(next_state, action)
        v_next_state = v[next_state]
        if next_state == env.final_state:
            break

    td_target_estimate = reward + math.pow(gamma, n) * v_next_state
    td_error = td_target_estimate - v[state]
    return v[state] + alpha * td_error


def td_n(env, n, episodes=1000, gamma=0.5, epsilon=0.9, alpha=0.05, epsilon_decay=0.99, updates=False):
    function = td_n_function
    return generic_value_algorithm(env, n, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)


def td_0_function(env, v, n, reward, state, next_state, return_, episode, gamma, alpha):
    td_target_estimate = reward + gamma * v[next_state]
    td_error = td_target_estimate - v[state]
    return v[state] + alpha * td_error


def td_0(env, episodes=1000, gamma=0.5, epsilon=0.9, alpha=0.05, epsilon_decay=0.99, updates=False):
    function = td_0_function
    return generic_value_algorithm(env, None, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)


def monte_carlo_function(env, v, n, reward, state, next_state, return_, episode, gamma, alpha):
    return v[state] + (1 / (episode + 1)) * (return_ + v[state])


def monte_carlo(env, episodes=1000, epsilon=0.9, epsilon_decay=0.99, updates=False):
    function = monte_carlo_function
    return generic_value_algorithm(env, None, function, episodes, None, epsilon, None, epsilon_decay, updates)
