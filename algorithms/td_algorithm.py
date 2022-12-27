import math
import numpy as np
import util

random = np.random.random
randint = np.random.randint


def get_pi_from_v(env, v):
    pi = dict()

    for state in env.all_possible_states:

        possible_actions, action_possible = env.get_possible_actions(state)
        if not action_possible:
            break

        # greedy
        # action = argmax of possible actions (reward + gamma*v[next state])
        int_best_actions = util.argmax_multi(
            [env.get_reward(env.get_next_state(state, a)) +
             v[env.get_next_state(state, a)]
             for a in possible_actions])

        best_actions = [env.action_to_str(possible_actions[int_best_action]) for int_best_action in int_best_actions]
        pi[state] = best_actions
    return pi


def td_0(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.01, epsilon_decay=0.99, updates=False, prev_state=False):
    fitness_curve = list()

    # Value States
    v = np.full(env.dimensions, 0, dtype=float)

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.get_start_state()

        return_ = 0
        prev_states = list()

        # if not final state
        while not env.done(state):

            # choose a possible action
            # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
            possible_actions, action_possible = env.get_possible_actions(state, prev_states)
            if len(possible_actions) == 0 or not action_possible:
                break

            # Step next state, here we use epsilon-greedy algorithm.
            if random() < epsilon:
                # choose random action
                action = possible_actions[randint(0, len(possible_actions))]
            else:
                # greedy
                # action = argmax of possible actions (reward + gamma*v[next state])
                action = possible_actions[util.argmax(
                    [env.get_reward(env.get_next_state(state, a)) +
                     gamma * v[env.get_next_state(state, a)]
                     for a in possible_actions])]

            next_state = env.get_next_state(state, action)

            reward = env.get_reward(tuple(next_state))
            if prev_state:
                prev_states.append(state)
            return_ += reward

            # Update V value
            td_target_estimate = reward + gamma * v[tuple(next_state)]
            td_error = td_target_estimate - v[state]
            v[state] = v[state] + alpha * td_error

            # Go to the next state
            state = next_state

        fitness_curve.append(return_)
        epsilon *= epsilon_decay

        # Display training progress
        if episode % 10 == 0 and updates:
            print("Training episode: %d" % episode)
            print(v)
            # print("     -       -       -")
            print(f"Current epsilon: {epsilon}")
            print(f"the return: {return_}")
            print("------------------------------------------------")

    return v, fitness_curve
