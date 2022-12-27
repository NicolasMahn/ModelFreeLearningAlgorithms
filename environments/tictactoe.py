import numpy as np
from .generic_environment import GenericEnvironment

random = np.random.random
randint = np.random.randint


class TicTacToe(GenericEnvironment):
    def __init__(self, player):
        super().__init__(dimensions=[3, 3, 3, 3, 3, 3, 3, 3, 3],
                         actions=[(player, 0, 0, 0, 0, 0, 0, 0, 0),  # a3
                                  (0, player, 0, 0, 0, 0, 0, 0, 0),  # b3
                                  (0, 0, player, 0, 0, 0, 0, 0, 0),  # c3
                                  (0, 0, 0, player, 0, 0, 0, 0, 0),  # a2
                                  (0, 0, 0, 0, player, 0, 0, 0, 0),  # b2
                                  (0, 0, 0, 0, 0, player, 0, 0, 0),  # c2
                                  (0, 0, 0, 0, 0, 0, player, 0, 0),  # a1
                                  (0, 0, 0, 0, 0, 0, 0, player, 0),  # b1
                                  (0, 0, 0, 0, 0, 0, 0, 0, player)],  # c1
                         start_state=(0, 0, 0,
                                      0, 0, 0,
                                      0, 0, 0),
                         actions_str=["a3", "b3", "c3", "a2", "b2", "c2", "a1", "b1", "c1"],
                         all_possible_states=self.__get_all_possible_states())
        self.player = player
        self.opponent = 3 - player
        pass

    player = int()
    opponent = int()

    def get_start_state(self):
        if self.player == 2:
            start_state = [0, 0, 0,
                           0, 0, 0,
                           0, 0, 0]
            start_state[randint(0, 8)] = self.opponent
            self.start_state = tuple(start_state)
        return self.start_state

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

    def get_possible_next_states(self, state, action):
        opponents_state = \
                    [action[i] + state[i] for i in range(len(action))]

        possible_actions, action_possible = \
                            self.get_possible_actions(opponents_state)
        if not action_possible:
            return opponents_state
        possible_next_states = list()
        for possible_action in possible_actions:
            possible_action = list(possible_action)
            if self.player == 1:
                possible_next_states.append(tuple(
                    [int(possible_action[i] * 2 + opponents_state[i])
                                for i in range(len(possible_action))]))
            else:
                possible_next_states.append(tuple(
                    [int(possible_action[i] / 2 + opponents_state[i])
                                for i in range(len(possible_action))]))
        return possible_next_states

    def get_next_state(self, state, action):
        possible_next_states = self.get_possible_next_states(state, action)
        # raise Exception("get_next_state was not properly implemented")
        if type(possible_next_states[0]) is int:
            return possible_next_states
        return possible_next_states[randint(0, len(possible_next_states))]


class TictactoeVS(GenericEnvironment):
    def __init__(self):

        self.player = 1
        self.opponent = 2

        dimensions = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        actions = [(self.player, 0, 0, 0, 0, 0, 0, 0, 0),  # a3
                   (0, self.player, 0, 0, 0, 0, 0, 0, 0),  # b3
                   (0, 0, self.player, 0, 0, 0, 0, 0, 0),  # c3
                   (0, 0, 0, self.player, 0, 0, 0, 0, 0),  # a2
                   (0, 0, 0, 0, self.player, 0, 0, 0, 0),  # b2
                   (0, 0, 0, 0, 0, self.player, 0, 0, 0),  # c2
                   (0, 0, 0, 0, 0, 0, self.player, 0, 0),  # a1
                   (0, 0, 0, 0, 0, 0, 0, self.player, 0),  # b1
                   (0, 0, 0, 0, 0, 0, 0, 0, self.player)]  # c1

        self.opponents_actions = [(self.opponent, 0, 0, 0, 0, 0, 0, 0, 0),  # a3
                                  (0, self.opponent, 0, 0, 0, 0, 0, 0, 0),  # b3
                                  (0, 0, self.opponent, 0, 0, 0, 0, 0, 0),  # c3
                                  (0, 0, 0, self.opponent, 0, 0, 0, 0, 0),  # a2
                                  (0, 0, 0, 0, self.opponent, 0, 0, 0, 0),  # b2
                                  (0, 0, 0, 0, 0, self.opponent, 0, 0, 0),  # c2
                                  (0, 0, 0, 0, 0, 0, self.opponent, 0, 0),  # a1
                                  (0, 0, 0, 0, 0, 0, 0, self.opponent, 0),  # b1
                                  (0, 0, 0, 0, 0, 0, 0, 0, self.opponent)]  # c1

        super().__init__(dimensions=dimensions,
                         actions=actions,
                         start_state=(0, 0, 0,
                                      0, 0, 0,
                                      0, 0, 0),
                         actions_str=["a3", "b3", "c3",
                                      "a2", "b2", "c2",
                                      "a1", "b1", "c1"],
                         all_possible_states=
                                self.__get_all_possible_states())

    player = int()
    opponent = int()

    opponents_actions = None

    def get_start_state(self):
        return self.start_state

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
    def get_next_state(action, state):
        return [action[i] + state[i] for i in range(len(action))]

    def get_possible_actions(self, state, prev_states=[], opponent=False):
        action_possible = False
        state = list(state)
        possible_actions = list()
        for i in range(len(state)):
            if state[i] == 0:
                if opponent:
                    possible_actions.append(self.opponents_actions[i])
                else:
                    possible_actions.append(self.actions[i])
                action_possible = True
        return possible_actions, action_possible

    def get_possible_qualities_and_actions(self, q_table, state, prev_states=[], opponent=False):
        possible_q = list()
        possible_actions, _ = self.get_possible_actions(state, opponent=opponent)
        for possible_action in possible_actions:
            possible_q.append(q_table[self.state_to_int(state), self.action_to_int(possible_action, opponent)])

        return possible_actions, possible_q

    def action_to_int(self, action, opponent=False):
        if opponent:
            return self.opponents_actions.index(action)
        return self.actions.index(action)

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

    def get_reward(self, state, opponent=False):
        _, winner = self._result(state)
        if winner == self.player:
            reward = 100
        elif winner == self.opponent:
            reward = -100
        else:
            reward = 0

        if opponent:
            return reward * -1
        else:
            return reward

    def get_winner(self, state):
        _, winner = self._result(state)
        return winner

