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
        expected_return = dict()
        for possible_action in possible_actions:
            possible_next_states = env.get_possible_next_states(state, possible_action)
            possible_rewards = list()
            possible_vs = list()
            if type(possible_next_states[0]) is int:
                possible_next_states = [[possible_next_states]]
            for possible_next_state in possible_next_states:
                possible_rewards.append(env.get_reward(tuple(possible_next_state)))
                possible_vs.append(v[tuple(possible_next_state)])
            expected_return[possible_action] = np.average(possible_rewards) + (np.average(possible_vs))
        int_best_actions = util.argmax_multi([expected_return[a] for a in possible_actions])

        best_actions = [env.action_to_str(possible_actions[int_best_action]) for int_best_action in int_best_actions]
        pi[state] = best_actions
    return pi


def generic_value_algorithm(env, n, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates):
    fitness_curve = list()

    # Value States
    v = np.full(env.dimensions, 0, dtype=float)

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.start_state

        prev_states = list()
        return_ = 0

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
                expected_return = dict()
                for possible_action in possible_actions:
                    possible_next_states = env.get_possible_next_states(state, possible_action)
                    possible_rewards = list()
                    possible_vs = list()
                    if type(possible_next_states[0]) is int:
                        possible_next_states = [[possible_next_states]]
                    for possible_next_state in possible_next_states:
                        possible_rewards.append(env.get_reward(tuple(possible_next_state)))
                        possible_vs.append(v[tuple(possible_next_state)])
                    expected_return[possible_action] = np.average(possible_rewards) + (gamma * np.average(possible_vs))
                action = possible_actions[util.argmax([expected_return[a] for a in possible_actions])]

            next_state = env.get_next_state(state, action)

            reward = env.get_reward(tuple(next_state))
            return_ += reward

            # Update V value
            v[state] = function(env, v, n, reward, state, next_state, gamma, alpha)

            # Go to the next state
            prev_states.append(state)
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

    return v, fitness_curve


def td_n_function(env, v, n, reward, state, next_state, gamma, alpha):
    returnN = reward
    v_next_state = v[next_state]
    for cn in range(1, n):
        possible_actions, action_possible = env.get_possible_actions(next_state)
        if not action_possible:
            break
        action = possible_actions[util.argmax([v[env.get_next_state(next_state, a)] for a in possible_actions])]
        returnN += math.pow(gamma, cn) * env.get_reward(env.get_next_state(next_state, action))
        next_state = env.get_next_state(next_state, action)
        v_next_state = v[next_state]
        if env.done(next_state):
            break

    td_target_estimate = reward + math.pow(gamma, n) * v_next_state
    td_error = td_target_estimate - v[state]
    return v[state] + alpha * td_error


def td_n(env, n, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.01, epsilon_decay=0.99, updates=False):
    function = td_n_function
    return generic_value_algorithm(env, n, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)


def td_0_function(env, v, n, reward, state, next_state, gamma, alpha):
    td_target_estimate = reward + gamma * v[tuple(next_state)]
    td_error = td_target_estimate - v[state]
    return v[state] + alpha * td_error


def td_0(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.01, epsilon_decay=0.99, updates=False):
    function = td_0_function
    return generic_value_algorithm(env, None, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)
