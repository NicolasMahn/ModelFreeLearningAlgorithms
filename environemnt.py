import numpy as np


class Labyrinth:
    start_state = (5, 0)
    final_state = (1, 2)

    # Labyrinth width
    width = 4
    # Labyrinth height
    height = 6

    # Reward
    _rewards = np.matrix([[-1, -1, -1, -1],
                          [-1, -np.inf, 60, -1],
                          [-1, -np.inf, -20, -np.inf],
                          [-1, -1, -1, -1],
                          [-70, -10, -np.inf, -1],
                          [-1, -1, -1, -1]])

    # Actions
    actions = [(-1, -1),  # NW
               (-1, 0),  # N
               (-1, 1),  # NE
               (0, -1),  # W
               (0, 1),  # E
               (1, -1),  # SW
               (1, 0),  # S
               (1, 1)]  # SE

    _actions_str = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]

    # since rewards are actually given for state action pairs
    def get_r(self, state, action):
        next_state = self.get_next_state(state, action)
        if next_state[0] >= self.height or next_state[0] < 0 or next_state[1] >= self.width or next_state[1] < 0:
            return -np.inf
        else:
            return self._rewards.item(next_state)

    def action_to_str(self, action):
        return self.int_action_to_str(self.actions.index(action))

    def int_action_to_str(self, action):
        return self._actions_str[action]

    # since state is a tuple in this example but not in the QTable
    def state_to_int(self, state):
        return state[0] * self.width + state[1]

    # since action is a tuple in this example but not in the QTable
    def action_to_int(self, action):
        return self.actions.index(action)

    @staticmethod
    def get_next_state(state, action):
        return state[0] + action[0], state[1] + action[1]

    def __int__(self):
        pass
