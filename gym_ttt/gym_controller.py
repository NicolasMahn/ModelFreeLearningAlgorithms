import gym
import gym_tictactoe
import numpy as np
import util

random = np.random.random
randint = np.random.randint

'''
Here some tests where conducted that didn't pan out as planed
'''


def play_game(actions, step_fn=input):
    env = gym.make('tictactoe-v0')
    env.reset()

    # Play actions in action profile
    for action in actions:
        print(env.step(action))
        env.render()
        if step_fn:
            step_fn()
    return env


def train(episodes=500, gamma=0.9, epsilon=0.4, alpha=0.1, epsilon_decay=0.99, updates=False):
    fitness_curve = list()
    env = gym.make('tictactoe-v0')
    q_table = np.zeros([3**3, 3**3])
    # the main training loop
    for episode in range(episodes + 1):

        # initial state
        state = env.reset()

        return_ = 0

        # if not final state
        while True:

            # choose a possible action

            # Step next state, here we use epsilon-greedy algorithm.
            if random() < epsilon:
                # choose random action
                rand = int(f"{str(randint(0, 3))}{str(randint(0, 3))}{str(randint(0, 3))}{str(randint(0, 3))}")
                print(rand)
                action = rand
            else:
                # greedy
                action = np.argmax(q_table[state])
                print(action)

            next_state, reward, done, info = env.step(f"{action:04d}")

            return_ += reward

            # Update Q value
            td_target_estimate = reward + gamma * q_table[next_state, :].max()
            td_error = td_target_estimate - q_table[state, action]
            q_table[state, action] = q_table[state, action] + alpha * td_error

            # Go to the next state
            state = next_state

            if done:
                break

        fitness_curve.append(return_)

        epsilon *= epsilon_decay

        # Display training progress
        if episode % 100 == 0 and updates:
            print("Training episode: %d" % episode)
            print(q_table)
            # print("     -       -       -")
            print(f"Current epsilon: {epsilon}")
            print(f"Info: {info}")
            print(f"the return: {return_}")
            print("------------------------------------------------")

    return q_table, fitness_curve


def test():
    actions = ['1021', '2111', '1221', '2222', '1121']
    _ = play_game(actions, None)
