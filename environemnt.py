import numpy as np


class Labyrinth:
    startState = (5, 0)
    finalState = (1, 2)

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

    _actionsStr = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]

    # since rewards are actually given for state action pairs
    def getR(self, state, action):
        nextState = self.getNextState(state, action)
        if nextState[0] >= self.height or nextState[0] < 0 or nextState[1] >= self.width or nextState[1] < 0:
            return -np.inf
        else:
            return self._rewards.item(nextState)

    def actionToStr(self, action):
        return self.intActionToStr(self.actions.index(action))

    def intActionToStr(self, action):
        return self._actionsStr[action]

    # since state is a tuple in this example but not in the QTable
    def stateToInt(self, state):
        return state[0] * self.width + state[1]

    # since action is a tuple in this example but not in the QTable
    def actionToInt(self, action):
        return self.actions.index(action)

    @staticmethod
    def getNextState(state, action):
        return state[0] + action[0], state[1] + action[1]

    def __int__(self):
        pass
