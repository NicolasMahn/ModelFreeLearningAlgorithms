import numpy as np


class Labyrinth:
    def __int__(self):
        pass

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


class TicTacToe:
    def __int__(self, player):
        self.player = player
        pass

    player = 1

    start_state = (0, 0, 0,
                   0, 0, 0,
                   0, 0, 0)

    # dimensions of state table
    dimensions = 9

    # depth of each dimension
    depth = [3, 3, 3, 3, 3, 3, 3, 3, 3]

    # Actions
    actions = [(player, 0, 0, 0, 0, 0, 0, 0, 0),  # a3
               (0, player, 0, 0, 0, 0, 0, 0, 0),  # a2
               (0, 0, player, 0, 0, 0, 0, 0, 0),  # a1
               (0, 0, 0, player, 0, 0, 0, 0, 0),  # b3
               (0, 0, 0, 0, player, 0, 0, 0, 0),  # b2
               (0, 0, 0, 0, 0, player, 0, 0, 0),  # b1
               (0, 0, 0, 0, 0, 0, player, 0, 0),  # c3
               (0, 0, 0, 0, 0, 0, 0, player, 0),  # c2
               (0, 0, 0, 0, 0, 0, 0, 0, player)]  # c1

    _actions_str = ["a3", "a2", "a1", "b3", "b2", "b1", "c3", "c2", "c1"]

    '''
    This function finds out if the game is over and who won if it is over
    '''

    @staticmethod
    def _result(state):
        is_done = False
        winner = 0

        # player won?
        for p in range(1, 3):
            a = state[0] == p and state[1] == p and state[2] == p
            b = state[3] == p and state[4] == p and state[5] == p
            c = state[6] == p and state[7] == p and state[8] == p

            d = state[0] == p and state[3] == p and state[6] == p
            e = state[1] == p and state[4] == p and state[7] == p
            f = state[2] == p and state[5] == p and state[8] == p

            g = state[0] == p and state[4] == p and state[8] == p
            h = state[2] == p and state[4] == p and state[6] == p

            if a or b or c or d or e or f or g or h:
                is_done = True
                winner = p

        # draw?
        if 0 not in state:
            is_done = True

        return is_done, winner

    # since rewards are actually given for state action pairs

    def get_r(self, state):
        _, winner = self._result(state)
        if winner == self.player:
            return 100
        elif winner != 0:
            return -100
        else:
            return 0

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


