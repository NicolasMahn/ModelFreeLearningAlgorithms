import numpy as np

random = np.random.random
randint = np.random.randint

class GenericEnvironment:

    # depth of each dimension
    def __init__(self, dimensions, actions, start_state, actions_str=None, all_possible_states=None):
        self.dimensions = dimensions
        self.actions = actions
        self.start_state = start_state
        if actions_str is None:
            for action in actions:
                self._actions_str.append(str(action))
        else:
            self._actions_str = actions_str
        if all_possible_states is None:
            self.all_possible_states = self.__get_all_possible_states()
        else:
            self.all_possible_states = all_possible_states

    # depth of each dimension
    dimensions = list()

    # actions
    actions = list()

    # starting state
    start_state = tuple()

    all_possible_states = list()

    # since state is a tuple in this example but not in the QTable
    def state_to_int(self, state):
        return self.all_possible_states.index(tuple(state))

    def int_state_to_tuple(self, int_state):
        return self.all_possible_states[int_state]

    def action_to_str(self, action):
        return self.int_action_to_str(self.actions.index(action))

    def int_action_to_str(self, action):
        return self._actions_str[action]

    # since action is a tuple in this example but not in the QTable
    def action_to_int(self, action):
        return self.actions.index(action)

    def __get_all_possible_states(self):
        state = np.full(len(self.dimensions), 0)
        _, all_possible_states = self.__depth_of_dimension(state, 0, list())
        return all_possible_states

    def __depth_of_dimension(self, state, i, all_possible_states):
        for j in range(self.dimensions[i]):
            state[i] = j
            if i + 1 < len(self.dimensions):
                state, all_possible_states = self.__depth_of_dimension(state, i + 1, all_possible_states)
            else:
                all_possible_states.append(tuple(state))
        return list(state), all_possible_states

    def get_reward(self, state):
        raise Exception("get_reward was not properly implemented")

    def get_next_state(self, state, action):
        raise Exception("get_next_state was not properly implemented")

    def get_possible_next_states(self, state, action):
        raise Exception("get_possible_next_states was not properly implemented")

    def get_possible_qualities_and_actions(self, q_table, state, prev_states=[]):
        raise Exception("get_possible_qualities_and_actions was not properly implemented")

    def done(self, state):
        raise Exception("done was not properly implemented")

    def get_possible_actions(self, state, prev_state=[]):
        raise Exception("get_possible_actions was not properly implemented")

    def action_possible(self, state, action):
        raise Exception("action_possible was not properly implemented")


class Labyrinth(GenericEnvironment):
    def __init__(self):
        super().__init__(dimensions=[6, 4],
                         actions=[(-1, -1),  # NW
                                  (-1, 0),  # N
                                  (-1, 1),  # NE
                                  (0, -1),  # W
                                  (0, 1),  # E
                                  (1, -1),  # SW
                                  (1, 0),  # S
                                  (1, 1)],  # SE
                         start_state=(5, 0),
                         actions_str=["NW", "N", "NE", "W", "E", "SW", "S", "SE"],
                         all_possible_states=self.__get_all_possible_states())
        pass

    @staticmethod
    def __get_all_possible_states():
        state = [0, 0]
        all_possible_states = list()
        for a in range(6):
            state[0] = a
            for b in range(4):
                state[1] = b
                all_possible_states.append(tuple(state))
        return all_possible_states

    final_state = (1, 2)

    def done(self, state):
        return self.final_state == state

    # Reward
    _rewards = np.matrix([[-1, -1, -1, -1],
                          [-1, -np.inf, 60, -1],
                          [-1, -np.inf, -20, -np.inf],
                          [-1, -1, -1, -1],
                          [-70, -10, -np.inf, -1],
                          [-1, -1, -1, -1]])

    # since rewards are actually given for state action pairs
    def get_reward(self, next_state):
        if next_state[0] >= self.dimensions[0] or next_state[0] < 0 \
                or next_state[1] >= self.dimensions[1] or next_state[1] < 0:
            return -np.inf
        else:
            return self._rewards.item(next_state)

    def get_next_state(self, state, action):
        return state[0] + action[0], state[1] + action[1]

    def get_possible_next_states(self, state, action):
        return [(state[0] + action[0], state[1] + action[1])]

    def get_possible_actions(self, state, prev_state=[]):
        possible_actions = [action for action in self.actions
                            if self.get_reward(self.get_next_state(state, action)) != -np.inf
                            and self.get_next_state(state, action) not in prev_state]
        action_possible = len(possible_actions) > 0
        return possible_actions, action_possible

    def get_possible_qualities_and_actions(self, q_table, state, prev_states=[]):
        possible_actions = []
        possible_q = []
        for action in self.actions:
            next_state = self.get_next_state(state, action)
            if self.get_reward(next_state) != -np.inf and next_state not in prev_states:
                possible_actions.append(action)
                possible_q.append(q_table[self.state_to_int(state), self.action_to_int(action)])

        return possible_actions, possible_q


class TicTacToe(GenericEnvironment):
    def __init__(self, player):
        super().__init__(dimensions=[3, 3, 3, 3, 3, 3, 3, 3, 3],
                         actions=[(player, 0, 0, 0, 0, 0, 0, 0, 0),  # a3
                                  (0, player, 0, 0, 0, 0, 0, 0, 0),  # a2
                                  (0, 0, player, 0, 0, 0, 0, 0, 0),  # a1
                                  (0, 0, 0, player, 0, 0, 0, 0, 0),  # b3
                                  (0, 0, 0, 0, player, 0, 0, 0, 0),  # b2
                                  (0, 0, 0, 0, 0, player, 0, 0, 0),  # b1
                                  (0, 0, 0, 0, 0, 0, player, 0, 0),  # c3
                                  (0, 0, 0, 0, 0, 0, 0, player, 0),  # c2
                                  (0, 0, 0, 0, 0, 0, 0, 0, player)],  # c1
                         start_state=(0, 0, 0,
                                      0, 0, 0,
                                      0, 0, 0),
                         actions_str=["a3", "a2", "a1", "b3", "b2", "b1", "c3", "c2", "c1"],
                         all_possible_states=self.__get_all_possible_states())
        self.player = player
        self.opponent = 3 - player
        pass

    player = 1
    opponent = 3 - player

    """
    defining this recursively increases the execution time by ~3min
    """

    @staticmethod
    def __get_all_possible_states():
        all_possible_states = list()
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for a in range(3):
            state[0] = a
            for b in range(3):
                state[1] = b
                for c in range(3):
                    state[2] = c
                    for d in range(3):
                        state[3] = d
                        for e in range(3):
                            state[4] = e
                            for f in range(3):
                                state[5] = f
                                for g in range(3):
                                    state[6] = g
                                    for h in range(3):
                                        state[7] = h
                                        for i in range(3):
                                            state[8] = i
                                            all_possible_states.append(tuple(state))
        return all_possible_states

    '''
    This function finds out if the game is over and who won if it is over
    '''

    def done(self, state):
        is_done, _ = self._result(state)
        return is_done

    @staticmethod
    def _result(state):
        is_done = False
        winner = 0
        if type(state[0]) is not int:
            state = state[0]
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

    def get_reward(self, state):
        _, winner = self._result(state)
        if winner == self.player:
            return 100
        elif winner == self.opponent:
            return -100
        else:
            return 0

    def get_possible_next_states(self, state, action):
        opponents_state = [action[i] + state[i] for i in range(len(action))]

        possible_actions, action_possible = self.get_possible_actions(opponents_state)
        if not action_possible:
            return opponents_state
        possible_next_states = list()
        for possible_action in possible_actions:
            possible_action = list(possible_action)
            if self.player == 1:
                possible_next_states.append(tuple(
                    [int(possible_action[i] * 2 + opponents_state[i]) for i in range(len(possible_action))]))
            else:
                possible_next_states.append(tuple(
                    [int(possible_action[i] / 2 + opponents_state[i]) for i in range(len(possible_action))]))
        return possible_next_states

    def get_next_state(self, state, action):
        possible_next_states = self.get_possible_next_states(state, action)
        # raise Exception("get_next_state was not properly implemented")
        if type(possible_next_states[0]) is int:
            return possible_next_states
        return possible_next_states[randint(0, len(possible_next_states))]

    def get_possible_actions(self, state, prev_state=[]):
        action_possible = False
        state = list(state)
        # state[self.action_to_int(action)] = self.player
        possible_actions = list()
        for i in range(len(state)):
            if state[i] == 0:
                possible_actions.append(self.actions[i])
                action_possible = True
        return possible_actions, action_possible

    def get_possible_qualities_and_actions(self, q_table, state, prev_states=[]):
        possible_q = list()
        possible_actions, _ = self.get_possible_actions(state)
        for possible_action in possible_actions:
            possible_q.append(q_table[self.state_to_int(state), self.action_to_int(possible_action)])

        return possible_actions, possible_q

    def action_possible(self, state, action):
        if state[self.action_to_int(action)] == 0:
            return True
        return False


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
        # self._get_all_possible_states()
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

    def get_possible_next_states(self, state, action):
        return [self.get_next_state(state, action)]

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
