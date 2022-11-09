import numpy as np
import numpy.random as random
import util


def get_pi(env, q_table):
    pi = {}
    state = (0, 0)

    while state[0] < env.height:
        while state[1] < env.width:

            if env.get_r(state, (0, 0)) == -np.inf:
                pi[state] = "-"
                state = (state[0], state[1] + 1)
                continue

            possible_actions, possible_q = get_possible_qualities_and_actions(env, q_table, state)
            max_quality = np.max(possible_q)
            best_actions = \
                [possible_actions[i] for i in range(0, len(possible_actions)) if possible_q[i] == max_quality]
            pi[state] = [env.action_to_str(ba) for ba in best_actions]

            state = (state[0], state[1] + 1)
        state = (state[0] + 1, 0)

    return pi


def get_possible_qualities_and_actions(env, q_table, state, prev_states=[]):
    # Even in random case, whose r[state, action] = -np.inf can't be chosen
    possible_actions = []
    possible_q = []
    for action in env.actions:
        if env.get_r(state, action) != -np.inf and env.get_next_state(state, action) not in prev_states:
            possible_actions.append(action)
            possible_q.append(q_table[env.state_to_int(state), env.action_to_int(action)])

    return possible_actions, possible_q


def generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates):
    fitness_curve = []

    # Q-Table
    q_table = np.full([env.height * env.width, len(env.actions)], 0, dtype=float)

    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.start_state
        prev_states = []

        return_ = 0

        # if not final state
        while state != env.final_state:
            int_state = env.state_to_int(state)

            # choose a possible action
            possible_actions, possible_q = get_possible_qualities_and_actions(env, q_table, state, prev_states)
            if len(possible_actions) == 0:
                break

            # Step next state, here we use epsilon-greedy algorithm.
            if random.random() < epsilon:
                # choose random action
                action = possible_actions[random.randint(0, len(possible_actions))]
            else:
                # greedy
                action = possible_actions[util.argmax(possible_q)]

            int_action = env.action_to_int(action)
            next_state = env.get_next_state(state, action)

            reward = env.get_r(state, action)
            return_ += reward

            # Update Q value
            q_table[int_state, int_action] = function(env, reward, q_table[int_state, int_action],
                                                      q_table[env.state_to_int(next_state), :], next_state,
                                                      gamma, alpha, epsilon)

            # Go to the next state
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

    return q_table, fitness_curve, get_pi(env, q_table)


def q_learning_function(env, reward, quality_state_action, quality_state, next_state, gamma, alpha, epsilon):
    TDTargetEstimate = reward + gamma * quality_state.max()
    TDError = TDTargetEstimate - quality_state_action
    return quality_state_action + alpha * TDError


def q_learning(env, episodes=1000, gamma=0.9, epsilon=0.9, alpha=0.05, epsilon_decay=0.99, updates=False):
    function = q_learning_function
    return generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)


def sarsa_function(env, reward, quality_state_action, quality_state, next_state, gamma, alpha, epsilon):
    # choose a possible action
    # Even in random case, we cannot choose actions whose r[state, action] = -np.inf.
    possible_actions = []
    possible_q = []
    for action in env.actions:
        if env.get_r(next_state, action) != -np.inf:
            possible_actions.append(action)
            possible_q.append(quality_state[env.action_to_int(action)])

    # Step next state, here we use epsilon-greedy algorithm.
    if random.random() < epsilon:
        # choose random action
        next_action = possible_actions[random.randint(0, len(possible_actions))]
    else:
        # greedy
        next_action = possible_actions[util.argmax(possible_q)]

    td_target_estimate = reward + gamma * quality_state[env.action_to_int(next_action)]
    td_error = td_target_estimate - quality_state_action
    return quality_state_action + alpha * td_error


def sarsa(env, episodes=1000, gamma=0.9, epsilon=0.9, alpha=0.05, epsilon_decay=0.99, updates=False):
    function = sarsa_function
    return generic_quality_algorithm(env, function, episodes, gamma, epsilon, alpha, epsilon_decay, updates)
