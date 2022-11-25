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


def monte_carlo_generic(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates):
    fitness_curve = list()  # for the graph

    # Value States
    v = np.full(env.dimensions, 0, dtype=float)

    # N
    n = np.full(env.dimensions, 0)

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.start_state

        prev_states = list()
        return_ = dict()

        # if not final state
        while not env.done(state):

            # Even in random case, not all actions are possible
            possible_actions, action_possible = env.get_possible_actions(state, prev_states)
            if not action_possible:
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

            # prepare for next state
            prev_states.append(state)
            n[state] += 1
            next_state = env.get_next_state(state, action)

            # collect reward
            return_[state] = env.get_reward(next_state)
            # Go to the next state
            state = next_state

        # Update V values
        v = function(v, return_, n, alpha)

        fitness_curve.append(sum(return_.values()))
        epsilon *= epsilon_decay

        # Display training progress
        if episode % 100 == 0 and updates:
            print("Training episode: %d" % episode)
            print(v)
            # print("     -       -       -")
            print(f"the return: {sum(return_.values())}")
            print("------------------------------------------------")

    return v, fitness_curve


def monte_carlo(env, episodes=500, gamma=0.9, epsilon=0.4, epsilon_decay=0.99, updates=False):
    function = monte_carlo_function
    return monte_carlo_generic(env, function, episodes, gamma, epsilon, None, epsilon_decay, updates)


def monte_carlo_function(v, return_, n, alpha):
    for state in return_:
        v[state] = v[state] + (1 / n[state]) * (return_[state] - v[state])
    return v


def monte_carlo_constant_alpha(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.01, epsilon_decay=0.99, updates=False):
    function = monte_carlo_const_alpha_function
    return monte_carlo_generic(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)


def monte_carlo_const_alpha_function(v, return_, n, alpha):
    for state in return_:
        v[state] = v[state] + alpha * (return_[state] - v[state])
    return v
