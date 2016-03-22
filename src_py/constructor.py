#! /usr/bin/env python

import numpy as np
import sys

class State(object):

    def __init__(self, term, s_index, prop_values):
        self._term = term
        self._s_index = s_index
        self._prop_values = prop_values
        self._name = self.prop_values_to_str()

    def prop_values_to_str(self): 
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

    def __init__(self, discount, prop_names): 
        self._discount = discount
        self._num_comp_states = 7
        self._prop_names = prop_names

        self.generate_state_set()
        self.generate_action_set()
        self.generate_observation_set()
        self.generate_trans_fun()
        self.load_confusion_matrix('../models/xval_predicate_behavior_confusion_matrices.csv')
        self.generate_obs_fun()
        self.generate_reward_fun()


    def generate_state_set(self):

        self._states = []

        depth = len(self._prop_names)
        for i in range(self._num_comp_states):
            self.generate_state_set_helper(i, 0, [], depth)

        self._states.append(State(True, None, None))

    def generate_state_set_helper(self, s_index, curr_depth, path, depth):
        
        if len(path) == depth:
            self._states.append(State(False, i, path))

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
            self._actions.append(Action(True, None, path))

        self.generate_action_set_helper(curr_depth+1, path.append('0'), depth)
        self.generate_action_set_helper(curr_depth+1, path.append('1'), depth)

    def generate_observation_set(self):

        self._observations = []
        self.generate_observation_set_helper(0, [], len(self._prop_name))
        self._observations.append(True, None)

    def generate_observation_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._observations.append(Obs(False, path))

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
            elif a_val._name == 'ask' or a_val._name == 'press':
                for s_idx, s_val in enumerate(self._states):
                    self._trans[a_idx, s_idx, s_idx] = 1.0
            elif a_val._name == 'push':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = get_state_index(False, 6, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            elif a_val._name == 'grasp':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = get_state_index(False, 2, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            elif a_val._name == 'lift':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 2:
                        tmp_s_idx = get_state_index(False, 3, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            elif a_val._name == 'hold':
                success_push = 0.99
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 3:
                        tmp_s_idx = get_state_index(False, 4, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            elif a_val._name == 'lower':
                success_push = 0.99
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 4:
                        tmp_s_idx = get_state_index(False, 5, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            elif a_val._name == 'drop':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 5:
                        tmp_s_idx = get_state_index(False, 6, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            elif a_val._name == 'reinit':
                success_push = 1.0
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 6:
                        tmp_s_idx = get_state_index(False, 0, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0
            elif a_val._term == True:
                for s_idx, s_val in enumerate(self._states):
                    self._trans[a_idx, s_idx, len(self._states)-1] = 1.0

    
    def load_confusion_matrix(self, path):
        f = open(path, 'r')
        lines = f.readlines()[1:]
        self.dic = {}
        for l in lines:
            words = l.split(',')
            if words[1] in self.dic:
                self.dic[words[1]][words[0]] = [int(w)+1 for w in words[2:]]
            else:
                self.dic[words[1]] = {words[0]: [int(w)+1 for w in words[2:]]}


    def generate_obs_fun(self):

        self._obs_fun = np.zeros((len(self._actions), len(self._states), len(self._observations)))
        for a_idx, a_val in enumerate(self._actions):
            for s_idx, s_val in enumerate(self._states):
                self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0

        for a_idx, a_val in enumerate(self._actions):
            for s_idx, s_val in enumerate(self._states):
                for o_idx, o_val in enumerate(self._observations):
                    prob = 1.0
                    for p_s_idx, p_s_val in enumerate(s_val._prop_values):
                        p_o_val = o_val._prop_values[p_s_idx]
                        mat = self.dic[a_val._name][self.prop_names[p_s_idx]]
                        if p_s_val == '0' and p_o_val == '0':
                            prob = prob * mat[3]/(mat[1] + mat[3])
                        elif p_s_val == '0' and p_o_val == '1':
                            prob = prob * mat[1]/(mat[1] + mat[3])
                        elif p_s_val == '1' and p_o_val == '0':
                            prob = prob * mat[2]/(mat[0] + mat[2])
                        elif p_s_val == '1' and p_o_val == '1':
                            prob = prob * mat[0]/(mat[0] + mat[2])
                    self._obs_fun[a_idx, s_idx, o_idx] = prob

    def generate_reward_fun(self):
        self._reward_fun = np.zeros((len(self._actions), len(self._states)))
        for a_idx, a_val in enumerate(self._actions):
            for s_idx, s_val in enumerate(self._states):
                if a_val.term == False:
                    self._reward_fun[a_idx, s_idx] = -2.0
                elif s_val.term == True:
                    self._reward_fun[a_idx, s_idx] = 0.0
                elif a_val._prop_values == s_val._prop_values:
                    self._reward_fun[a_idx, s_idx] = 100.0
                else:
                    self._reward_fun[a_idx, s_idx] = -100.0

    def write_to_file(self, path):
        
        s = 'discount: ' + str(self._discount) + '\nvalues: reward\n'
        s = 'states: '
        for state in self._states:
            s += state._name + ' '
        s += '\n\n'
        s = 'actions: '
        for action in self._actions:
            s += action._name + ' '
        s += '\n\n'
        s = 'observations: '
        for observation in self._observations:
            s += observation._name + ' '
        s += '\n\n'

        for a in range(len(self._actions)):
            s += 'T: ' + a._name + '\n'
            for s1 in range(len(self._states)):
                for s2 in range(len(self._states)):
                    s += str(self._trans[a, s1, s2]) + ' '
                s += '\n'
            s += '\n'

        for a in range(len(self._actions)):
            s += 'O: ' + a._name + '\n'
            for s1 in range(len(self._states)):
                for o in range(len(self._observations)):
                    s += str(self._obs_fun[a, s1, o]) + ' '
                s += '\n'
            s += '\n'

        for a in range(len(self._actions)):
            for s1 in range(len(self._states)):
                s += 'R: ' + a._name + ' : ' + s1._name + ' : * : * '
                s += str(self._reward_fun[a, s1]) + '\n'

        f = open(path, 'w')
        f.write(s)

def main(argv):
    
    model = Model(0.95, ['light', 'talk'])
    model.write_to_file('model.POMDP')

if __name__ == "__main__":
    main(sys.argv)
        























