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


def get_possible_actions(env, state, prev_states=[]):
    return [action for action in env.actions if env.get_r(state, action) != -np.inf
            and env.get_next_state(state, action) not in prev_states]


def monte_carlo(env, episodes=1000, epsilon=0.9, epsilon_decay=0.99, updates=False):
    fitness_curve = []

    # Value States
    v = np.full([env.height, env.width], 0, dtype=float)

    # N
    n = np.full([env.height, env.width], 0)

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.start_state
        prev_states = []

        return_ = {}

        # if not final state
        while state != env.final_state:

            # choose a possible action
            # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
            possible_actions = get_possible_actions(env, state, prev_states)
            if len(possible_actions) == 0:
                break

            # Step next state, here we use epsilon-greedy algorithm.
            if random.random() < epsilon:
                # choose random action
                action = possible_actions[random.randint(0, len(possible_actions))]
            else:
                # greedy
                action = possible_actions[util.argmax([v[env.get_next_state(state, a)] for a in possible_actions])]

            reward = env.get_r(state, action)
            return_[state] = reward
            n[state] += 1

            # Go to the next state
            prev_states.append(state)
            state = env.get_next_state(state, action)

        # Update V value
        v = monte_carlo_function(v, return_, n)

        fitness_curve.append(sum(return_.values()))
        epsilon *= epsilon_decay

        # Display training progress
        if episode % 100 == 0 and updates:
            print("Training episode: %d" % episode)
            print(v)
            # print("     -       -       -")
            print(f"the return: {sum(return_.values())}")
            print("------------------------------------------------")

    return v, fitness_curve, get_pi(env, v)


def monte_carlo_function(v, return_of_s, n):
    for state in return_of_s:
        v[state] = v[state] + (1 / (n[state])) * (return_of_s[state] - v[state])
    return v



