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
        if self._term is True:
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

    def __init__(self, discount, prop_names, high_acc, ask_cost): 
        self._discount = discount
        self._num_comp_states = 6
        self._prop_names = prop_names
        self._high = high_acc
        self._ask_cost = ask_cost

        self._states = []
        self._actions = []
        self._observations = []

        self.generate_state_set()
        self.generate_action_set()
        self.generate_observation_set()

        self._trans = np.zeros((len(self._actions), len(self._states), len(self._states)))
        self._obs_fun = np.zeros((len(self._actions), len(self._states), len(self._observations)))
        self._reward_fun = np.zeros((len(self._actions), len(self._states)))

        self.generate_trans_fun()
        self.load_confusion_matrix('../data/icra2014/confusion_matrices_train5.csv')
        self.generate_obs_fun()
        self.generate_reward_fun()


    def generate_state_set(self):

        for i in range(self._num_comp_states):
            self.generate_state_set_helper(i, 0, [], len(self._prop_names))

        self._states.append(State(True, None, None))

        # print(str([s._name for s in self._states]))
        # exit(1)

    def generate_state_set_helper(self, s_index, curr_depth, path, depth):
        
        # print('s_index: ' + str(s_index))
        # print('curr_depth: ' + str(curr_depth))
        # print('path: ' + str(path))
        # print('depth: ' + str(depth))
        # print
        if len(path) == depth:
            self._states.append(State(False, s_index, path))
            return

        self.generate_state_set_helper(s_index, curr_depth+1, path+['0'], depth)
        self.generate_state_set_helper(s_index, curr_depth+1, path+['1'], depth)

    def get_state_index(self, term, s_index, prop_values):
        if term == True:
            return len(self._states) - 1

        else:
            return s_index*pow(2, len(prop_values)) + int(''.join(prop_values), 2)

    def get_action_name(self, a_index):
        for a_idx, a_val in enumerate(self._actions):
            if a_index == a_idx:
                return a_val._name
        else:
            return ""


    def generate_action_set(self):

        # the action names must match the action names in confusion matrices (csv file)
        self._actions.append(Action(False, 'look', None))
        self._actions.append(Action(False, 'grasp', None))
        self._actions.append(Action(False, 'lift_slow', None))
        self._actions.append(Action(False, 'hold', None))
        self._actions.append(Action(False, 'shake', None))
        self._actions.append(Action(False, 'high_velocity_shake', None))
        self._actions.append(Action(False, 'low_drop', None))
        self._actions.append(Action(False, 'tap', None))
        self._actions.append(Action(False, 'poke', None))
        self._actions.append(Action(False, 'push', None))
        self._actions.append(Action(False, 'crush', None))

        self._actions.append(Action(False, 'ask', None))
        self._actions.append(Action(False, 'reinit', None))

        self.generate_action_set_helper(0, [], len(self._prop_names))

        # print(str([a._name for a in self._actions]))
        # exit(1)

    def generate_action_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._actions.append(Action(True, None, path))
            return

        self.generate_action_set_helper(curr_depth+1, path+['0'], depth)
        self.generate_action_set_helper(curr_depth+1, path+['1'], depth)

    def generate_observation_set(self):


        self.generate_observation_set_helper(0, [], len(self._prop_names))
        self._observations.append(Obs(True, None))

        # print(str([o._name for o in self._observations]))
        # exit(1)

    def generate_observation_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._observations.append(Obs(False, path))
            return

        self.generate_observation_set_helper(curr_depth+1, path+['0'], depth)
        self.generate_observation_set_helper(curr_depth+1, path+['1'], depth)

    def generate_trans_fun(self):

        # going through all actions based on their names
        for a_idx, a_val in enumerate(self._actions):

            if a_val._name == 'look':
                for s_idx, s_val in enumerate(self._states):

                    if s_val._term == False and s_val._s_index == 0:
                        tmp_s_idx = self.get_state_index(False, 1, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = 1.0
                    elif s_val._term == False and s_val._s_index == 1:
                        self._trans[a_idx, s_idx, len(self._states)-1] = 1.0
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0

            elif a_val._name == 'ask' or a_val._name == 'crush':
                for s_idx, s_val in enumerate(self._states):
                    self._trans[a_idx, s_idx, s_idx] = 1.0

            elif a_val._name == 'tap' or a_val._name == 'poke' or a_val._name == 'push':
                success_push = 0.9
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = self.get_state_index(False, 5, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0

            elif a_val._name == 'grasp':
                success_push = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 1:
                        tmp_s_idx = self.get_state_index(False, 2, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0

            elif a_val._name == 'lift_slow':
                success_push = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 2:
                        tmp_s_idx = self.get_state_index(False, 3, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    elif s_val._term == False and s_val._s_index == 2:
                        self._trans[a_idx, s_idx, len(self._states)-1] = 1.0                        
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0

            elif a_val._name == 'hold':
                success_push = 0.99
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 3:
                        tmp_s_idx = self.get_state_index(False, 4, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0

            # elif a_val._name == 'lower':
            #     success_push = 0.99
            #     for s_idx, s_val in enumerate(self._states):
            #         if s_val._term == False and s_val._s_index == 4:
            #             tmp_s_idx = self.get_state_index(False, 5, s_val._prop_values)
            #             self._trans[a_idx, s_idx, tmp_s_idx] = success_push
            #             self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
            #         else:
            #             self._trans[a_idx, s_idx, s_idx] = 1.0


            elif a_val._name == 'shake' or a_val._name == 'high_velocity_shake':
                success_push = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 4:
                        tmp_s_idx = self.get_state_index(False, 5, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = 1.0 - success_push
                        self._trans[a_idx, s_idx, s_idx] = success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0


            elif a_val._name == 'low_drop':
                success_push = 0.95
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 4:
                        tmp_s_idx = self.get_state_index(False, 5, s_val._prop_values)
                        self._trans[a_idx, s_idx, tmp_s_idx] = success_push
                        self._trans[a_idx, s_idx, s_idx] = 1.0 - success_push
                    else:
                        self._trans[a_idx, s_idx, s_idx] = 1.0

            elif a_val._name == 'reinit':
                success_push = 1.0
                for s_idx, s_val in enumerate(self._states):
                    if s_val._term == False and s_val._s_index == 5:
                        tmp_s_idx = self.get_state_index(False, 0, s_val._prop_values)
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

        # for a_idx, a_val in enumerate(self._actions):
        #     for s_idx, s_val in enumerate(self._states):
        #         self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0

        for a_idx, a_val in enumerate(self._actions):
            for s_idx, s_val in enumerate(self._states):

                if a_val._term == True or a_val._name == 'reinit' or s_val._term == True:
                    self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                    continue

                for o_idx, o_val in enumerate(self._observations):

                    prob = 1.0
                    if o_val._nonavail == True:
                        # self._obs_fun[a_idx, s_idx, o_idx] = prob
                        continue 

                    if a_val._name == 'ask':
                        if s_val._prop_values == o_val._prop_values:
                            self._obs_fun[a_idx, s_idx, o_idx] = self._high
                        else:
                            self._obs_fun[a_idx, s_idx, o_idx] = \
                            (1.0 - self._high)/(len(self._observations)-2.0)
                        continue 

                    if a_val._name == 'look' or a_val._name == 'press':
                        if s_val._s_index != 1:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'push' or a_val._name == 'poke' or a_val._name == 'tap':
                        if s_val._s_index != 5:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'grasp':
                        if s_val._s_index != 2:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'lift_slow':
                        if s_val._s_index != 3:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    elif a_val._name == 'hold':
                        if s_val._s_index != 4:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue
                    # elif a_val._name == 'lower':
                    #     if s_val._s_index != 5:
                    #         self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                    #         continue
                    elif a_val._name == 'low_drop' or a_val._name == 'shake' or a_val._name == 'high_velocity_shake':
                        if s_val._s_index != 5:
                            self._obs_fun[a_idx, s_idx, len(self._observations)-1] = 1.0
                            continue

                    for p_s_idx, p_s_val in enumerate(s_val._prop_values):

                        p_o_val = o_val._prop_values[p_s_idx]
                        mat = self.dic[a_val._name][self._prop_names[p_s_idx]]
                        if p_s_val == '0' and p_o_val == '0':
                            prob = prob * mat[3]/(mat[1] + mat[3])
                        elif p_s_val == '0' and p_o_val == '1':
                            prob = prob * mat[1]/(mat[1] + mat[3])
                        elif p_s_val == '1' and p_o_val == '0':
                            prob = prob * mat[2]/(mat[0] + mat[2])
                        elif p_s_val == '1' and p_o_val == '1':
                            prob = prob * mat[0]/(mat[0] + mat[2])

                    self._obs_fun[a_idx, s_idx, o_idx] = prob
                    # print('prob: ' + str(prob))

    def generate_reward_fun(self):

        for a_idx, a_val in enumerate(self._actions):
            for s_idx, s_val in enumerate(self._states):
                if s_val._term == True:
                    self._reward_fun[a_idx, s_idx] = 0.0
                elif a_val._term == False and a_val._name == 'ask':
                    self._reward_fun[a_idx, s_idx] = self._ask_cost

                elif a_val._term == False and a_val._name == 'look':
                    self._reward_fun[a_idx, s_idx] = -0.5
                elif a_val._term == False and a_val._name == 'grasp':
                    self._reward_fun[a_idx, s_idx] = -4.5
                elif a_val._term == False and a_val._name == 'lift_slow':
                    self._reward_fun[a_idx, s_idx] = -2.6              
                elif a_val._term == False and a_val._name == 'hold':
                    self._reward_fun[a_idx, s_idx] = -1.0
                elif a_val._term == False and a_val._name == 'shake':
                    self._reward_fun[a_idx, s_idx] = -3.8
                elif a_val._term == False and a_val._name == 'high_velocity_shake':
                    self._reward_fun[a_idx, s_idx] = -4.1
                elif a_val._term == False and a_val._name == 'low_drop':
                    self._reward_fun[a_idx, s_idx] = -2.6
                elif a_val._term == False and a_val._name == 'tap':
                    self._reward_fun[a_idx, s_idx] = -4.2
                elif a_val._term == False and a_val._name == 'poke':
                    self._reward_fun[a_idx, s_idx] = -4.1
                elif a_val._term == False and a_val._name == 'push':
                    self._reward_fun[a_idx, s_idx] = -4.5
                elif a_val._term == False and a_val._name == 'crush':
                    self._reward_fun[a_idx, s_idx] = -5.2

                elif a_val._prop_values == s_val._prop_values:
                    self._reward_fun[a_idx, s_idx] = 100.0
                else:
                    self._reward_fun[a_idx, s_idx] = -100.0

    def write_to_file(self, path):
        
        s = 'discount: ' + str(self._discount) + '\nvalues: reward\n\n'
        s += 'states: '
        for state in self._states:
            s += state._name + ' '
        s += '\n\n'
        s += 'actions: '
        for action in self._actions:
            s += action._name + ' '
        s += '\n\n'
        s += 'observations: '
        for observation in self._observations:
            s += observation._name + ' '
        s += '\n\n'

        for a in range(len(self._actions)):
            s += 'T: ' + self._actions[a]._name + '\n'
            for s1 in range(len(self._states)):
                for s2 in range(len(self._states)):
                    s += str(self._trans[a, s1, s2]) + ' '
                s += '\n'
            s += '\n'

        for a in range(len(self._actions)):
            s += 'O: ' + self._actions[a]._name + '\n'
            for s1 in range(len(self._states)):
                for o in range(len(self._observations)):
                    s += str(self._obs_fun[a, s1, o]) + ' '
                s += '\n'
            s += '\n'

        for a in range(len(self._actions)):
            for s1 in range(len(self._states)):
                s += 'R: ' + self._actions[a]._name + ' : ' + self._states[s1]._name + ' : * : * '
                s += str(self._reward_fun[a, s1]) + '\n'

        f = open(path, 'w')
        f.write(s)

def main(argv):
    
    model = Model(0.99, ['green', 'heavy'], 0.9, -90.0)
    model.write_to_file('model.pomdp')

if __name__ == "__main__":
    main(sys.argv)
        



