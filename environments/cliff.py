import numpy as np
from .generic_environment import GenericEnvironment

random = np.random.random
randint = np.random.randint


class Cliff(GenericEnvironment):
    def __init__(self):
        super().__init__(dimensions=[4, 9],
                         actions=[(-1, -1),  # NW
                                  (-1, 0),  # N
                                  (-1, 1),  # NE
                                  (0, -1),  # W
                                  (0, 1),  # E
                                  (1, -1),  # SW
                                  (1, 0),  # S
                                  (1, 1)],  # SE
                         start_state=(3, 0),
                         actions_str=["NW", "N", "NE", "W", "E", "SW", "S", "SE"],
                         all_possible_states=self.__get_all_possible_states())
        pass

    final_states = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]

    def done(self, state):
        return state in self.final_states

    # Reward
    _rewards = np.matrix([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                          [-1, -100, -100, -100, -100, -100, -100, -100, 100]])

    @staticmethod
    def __get_all_possible_states():
        state = [0, 0]
        all_possible_states = list()
        for a in range(4):
            state[0] = a
            for b in range(9):
                state[1] = b
                all_possible_states.append(tuple(state))
        return all_possible_states

    # since rewards are actually given for state action pairs
    def get_reward(self, state):
        if state[0] >= self.dimensions[0] or state[0] < 0 or state[1] >= self.dimensions[1] or state[1] < 0:
            return -np.inf
        else:
            return self._rewards.item(state)

    def get_next_state(self, state, action):
        return state[0] + action[0], state[1] + action[1]

    def get_possible_actions(self, state, prev_state=[]):
        possible_actions = [action for action in self.actions if self.get_reward(self.get_next_state(state, action))
                            != -np.inf and self.get_next_state(state, action) not in prev_state]
        return possible_actions, True

    def get_possible_qualities_and_actions(self, q_table, state, prev_states=[]):
        # Even in random case, whose r[state, action] = -np.inf can't be chosen
        possible_actions = []
        possible_q = []
        for action in self.actions:
            next_state = self.get_next_state(state, action)
            if self.get_reward(next_state) != -np.inf and next_state not in prev_states:
                possible_actions.append(action)
                possible_q.append(q_table[self.state_to_int(state), self.action_to_int(action)])

        return possible_actions, possible_q

