import numpy as np
import util

random = np.random.random
randint = np.random.randint


def get_pi_from_q(env, q_table):
    pi = dict()

    for state in env.all_possible_states:

        possible_actions, possible_q = env.get_possible_qualities_and_actions(q_table, state)
        if len(possible_actions) == 0:
            break

        # greedy
        int_best_actions = util.argmax_multi(possible_q)

        best_actions = [env.action_to_str(possible_actions[int_best_action]) for int_best_action in int_best_actions]
        pi[state] = best_actions
    return pi


def get_v_from_q(env, q_table):
    v = dict()

    for state in env.all_possible_states:
        best_state = q_table[env.state_to_int(state), :].max()

        v[state] = best_state
    return v


def generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates, prev_state):
    fitness_curve = list()

    # Q-Table
    q_table = np.full([np.prod(env.dimensions), len(env.actions)], 0, dtype=float)

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.get_start_state()

        prev_states = list()
        return_ = 0

        # if not final state
        while not env.done(state):
            int_state = env.state_to_int(state)

            # choose a possible action
            possible_actions, possible_q = env.get_possible_qualities_and_actions(q_table, state, prev_states)
            if len(possible_actions) == 0:
                break

            # Step next state, here we use epsilon-greedy algorithm.
            if random() < epsilon:
                # choose random action
                action = possible_actions[randint(0, len(possible_actions))]
            else:
                # greedy
                action = possible_actions[util.argmax(possible_q)]
            int_action = env.action_to_int(action)

            next_state = env.get_next_state(state, action)

            reward = env.get_reward(next_state)
            return_ += reward

            # Update Q value
            q_table[int_state, int_action] = function(env, q_table, reward, state, action, next_state,
                                                      gamma, alpha, epsilon)

            # Go to the next state
            if prev_state:
                prev_states.append(state)
            state = next_state

        fitness_curve.append(return_)

        epsilon *= epsilon_decay

        # Display training progress
        if episode % 100 == 0 and updates:
            print("Training episode: %d" % episode)
            print(q_table)
            # print("     -       -       -")
            print(f"Current epsilon: {epsilon}")
            print(f"the return: {return_}")
            print("------------------------------------------------")

    return q_table, fitness_curve


def q_learning_function(env, q_table, reward, state, action, next_state, gamma, alpha, epsilon):
    td_target_estimate = reward + gamma * q_table[env.state_to_int(next_state), :].max()
    td_error = td_target_estimate - q_table[env.state_to_int(state), env.action_to_int(action)]
    return q_table[env.state_to_int(state), env.action_to_int(action)] + alpha * td_error


def q_learning(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.1, epsilon_decay=0.99, updates=False,
               prev_state=False):
    function = q_learning_function
    return generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates, prev_state)


def sarsa_function(env, q_table, reward, state, action, next_state, gamma, alpha, epsilon):
    # choose a possible action
    # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
    possible_actions, possible_q = env.get_possible_qualities_and_actions(q_table, next_state)
    if len(possible_actions) == 0:
        return q_table[env.state_to_int(state), env.action_toint(action)] + alpha * \
            (reward - q_table[env.state_to_int(state), env.action_to_int(action)])

    # Step next state, here we use epsilon-greedy algorithm.
    if random() < epsilon:
        # choose random action
        next_action = possible_actions[randint(0, len(possible_actions))]
    else:
        # greedy
        next_action = possible_actions[util.argmax(possible_q)]

    td_target_estimate = reward + gamma * q_table[env.state_to_int(next_state), env.action_to_int(next_action)]
    td_error = td_target_estimate - q_table[env.state_to_int(state), env.action_to_int(action)]
    return q_table[env.state_to_int(state), env.action_to_int(action)] + alpha * td_error


def sarsa(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.1, epsilon_decay=0.99, updates=False, prev_state=False):
    function = sarsa_function
    return generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates, prev_state)
