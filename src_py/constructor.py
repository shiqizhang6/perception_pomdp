#! /usr/bin/env python

import numpy as np

class State(object):

    def __init__(self, term, s_index, prop_values):
        self._term = term
        self._s_index = s_index
        self._prop_values = prop_values
        self._name = self.prop_values_to_str()

    def prop_values_to_str(self)
        if term is True:
            return 'terminal'
        else:
            return 's'+str(self._s_index)+'p'+''.join(self._prop_values)

class Action(object):

    def __init__(self, term, name, prop_values):
        self._term = term

        if term == False:
            self._prop_values = None
            self._name = name
        else:
            self._prop_values = prop_values
            self._name = 'a'+''.join(prop_values)


class Obs(object):

    def __init__(self, nonavail, prop_values):
        self._nonavail = nonavail
        if nonavail == False:
            self._prop_values = prop_values
            self._name = 'p'+''.join(prop_values)
        else:
            self._prop_values = None
            self._name = 'na'


class Model:

    def __init__(self, discount, num_comp_states): 
        self._discount = discount
        self._num_comp_states = num_comp_states

    def get_props(self):
        self.prop_names = []
        self.prop_probs = []

        #TODO: add code to extract name+value information from input

        assert len(prop_probs) == 2
        for i in range(len(prop_probs):
            assert len(prop_names) == len(prop_probs[i])
            assert prop_probs[i][0] + prop_probs[i][1] == 1.0
    
    def generate_state_set(self):

        self._states = []

        depth = len(self._prop_names)
        for i in range(self._num_comp_states):
            self.generate_state_set_helper(i, 0, [], depth)

        self._states.append(State(True, None, None)

    def generate_state_set_helper(self, s_index, curr_depth, path, depth):
        
        if len(path) == depth:
            self._states.append(State(False, i, path)

        self.generate_state_set_helper(s_index, curr_depth+1, path.append('0'), depth)
        self.generate_state_set_helper(s_index, curr_depth+1, path.append('1'), depth)

    def get_state_index(self, term, s_index, prop_values):
        if term == True:
            return len(self._states) - 1

        else:
            return s_index*pow(2, len(prop_values)) + int(''.join(path), 2) - 1

    def generate_action_set(self):

        self._actions = []
        self._actions.append(Action(False, 'look', None))
        self._actions.append(Action(False, 'ask', None))
        self._actions.append(Action(False, 'press', None))
        self._actions.append(Action(False, 'push', None))
        self._actions.append(Action(False, 'grasp', None))
        self._actions.append(Action(False, 'lift', None))
        self._actions.append(Action(False, 'hold', None))
        self._actions.append(Action(False, 'lower', None))
        self._actions.append(Action(False, 'drop', None))
        self._actions.append(Action(False, 'reinit', None))

        self.generate_action_set_helper(0, [], len(self._prop_names))

    def generate_action_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._actions.append(Action(True, None, path)

        self.generate_action_set_helper(curr_depth+1, path.append('0'), depth)
        self.generate_action_set_helper(curr_depth+1, path.append('1'), depth)

    def generate_observation_set(self):

        self._observations = []
        self.generate_observation_set_helper(0, [], len(self._prop_name))
        self._observations.append(True, None)

    def generate_observation_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._observations.append(Obs(False, path)

        self.generate_observation_set_helper(curr_depth+1, path.append('0'), depth)
        self.generate_observation_set_helper(curr_depth+1, path.append('1'), depth)

    def generate_trans_fun(self):

        self._trans = np.zeros((len(self._actions), len(self._states), len(self._states)))

        for a_idx, a_val in enumerate(self._actions):
            if a_val._name == 'look':
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 0:
                        tmp_s_idx = get_state_index(False, 1, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = 1.0
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._name == 'ask' or a_val._name == 'press':
                for s_idx, s_val in enumerate(self._states):
                    self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._name == 'push':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = get_state_index(False, 6, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._name == 'grasp':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = get_state_index(False, 2, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._name == 'lift':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 2:
                        tmp_s_idx = get_state_index(False, 3, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._name == 'hold':
                success_push = 0.99
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 3:
                        tmp_s_idx = get_state_index(False, 4, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._name == 'lower':
                success_push = 0.99
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 4:
                        tmp_s_idx = get_state_index(False, 5, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._name == 'drop':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 5:
                        tmp_s_idx = get_state_index(False, 6, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._name == 'reinit':
                success_push = 1.0
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 6:
                        tmp_s_idx = get_state_index(False, 0, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            else if a_val._term == True:
                for s_idx, s_val in enumerate(self._states):
                    self._trans[a_idx, s_idx, len(self._states)-1] = 1.0

    def generate_obs_fun(self):

        self._obs_fun = np.zeros((len(self._actions), len(self._states), len(self._observations)))
        for a_idx, a_val in enumerate(self._actions):
            for s_idx, s_val in enumerate(self._states):
                self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0














