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
        return self.int_action_to_str(self.action_to_int(action))

    def int_action_to_str(self, int_action):
        return self._actions_str[int_action]

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

    def get_reward(self, next_state):
        raise Exception("get_reward was not properly implemented")

    def get_next_state(self, state, action):
        raise Exception("get_next_state was not properly implemented")

    def get_possible_qualities_and_actions(self, q_table, state, prev_states=[]):
        raise Exception("get_possible_qualities_and_actions was not properly implemented")

    def done(self, state):
        raise Exception("done was not properly implemented")

    def get_possible_actions(self, state, prev_state=[]):
        raise Exception("get_possible_actions was not properly implemented")

    def action_possible(self, state, action):
        if state[self.action_to_int(action)] == 0:
            return True
        return False

    def get_start_state(self):
        return self.start_state
