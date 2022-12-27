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

        possible_actions, possible_q = env.get_possible_qualities_and_actions(q_table, state)
        if len(possible_actions) == 0:
            break

        best_state = possible_q[env.state_to_int(state), :].max()

        v[state] = best_state
    return v


def generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates):
    winners = list()

    # Q-Table
    q_table = np.full([np.prod(env.dimensions), len(env.actions)], 0, dtype=float)

    state_player = None
    state_opponent = None
    action_player = None
    action_opponent = None

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state_player = env.get_start_state()

        while True:

            # player
            action_player = take_action(env, q_table, state_player, epsilon)

            # hand over to opponent
            next_state_opponent = env.get_next_state(state_player, action_player)

            if state_opponent is not None and action_opponent is not None:
                q_table[env.state_to_int(state_opponent), env.action_to_int(action_opponent, opponent=True)] = \
                    function(env, q_table, env.get_reward(next_state_opponent, opponent=True),
                             state_opponent, action_opponent, next_state_opponent, list(),
                             gamma, alpha, epsilon, opponent=True)

            state_opponent = next_state_opponent

            # if final state
            if env.done(state_opponent):
                final_state = state_opponent
                break

            action_opponent = take_action(env, q_table, state_opponent, epsilon, opponent=True)

            # hand over to player
            next_state_player = env.get_next_state(state_opponent, action_opponent)

            if state_player is not None and action_player is not None:
                q_table[env.state_to_int(state_player), env.action_to_int(action_player)] = \
                    function(env, q_table, env.get_reward(next_state_player),
                             state_player, action_player, next_state_player, list(),
                             gamma, alpha, epsilon)

            state_player = next_state_player

            # if final state
            if env.done(state_player):
                final_state = state_player
                break

        epsilon *= epsilon_decay
        winners.append(env.get_winner(final_state))

    return q_table, winners


def take_action(env, q_table, state, epsilon, opponent=False):
    # choose a possible action
    possible_actions, possible_q = env.get_possible_qualities_and_actions(q_table, state, opponent=opponent)
    if len(possible_actions) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Step next state, here we use epsilon-greedy algorithm.
    if random() < epsilon:
        # choose random action
        return possible_actions[randint(0, len(possible_actions))]
    else:
        # greedy
        return possible_actions[util.argmax(possible_q)]


def q_learning_function(env, q_table, reward, state, action, next_state, prev_states, gamma, alpha, epsilon,
                        opponent=False):
    td_target_estimate = reward + gamma * q_table[env.state_to_int(next_state), :].max()
    td_error = td_target_estimate - q_table[env.state_to_int(state), env.action_to_int(action, opponent=opponent)]
    return q_table[env.state_to_int(state), env.action_to_int(action, opponent=opponent)] + alpha * td_error


def q_learning_vs(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.1, epsilon_decay=0.99, updates=False):
    function = q_learning_function
    return generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)


def sarsa_function(env, q_table, reward, state, action, next_state, prev_states, gamma, alpha, epsilon):
    # choose a possible action
    # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
    possible_actions, possible_q = env.get_possible_qualities_and_actions(q_table, next_state, prev_states)
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


def sarsa_vs(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.1, epsilon_decay=0.99, updates=False):
    function = sarsa_function
    return generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)
