#! /usr/bin/env python


import random
import numpy as np
import sys
from constructor import Model, State, Action, Obs
from policy import Policy, Solver

class Simulator(object):

    def __init__(self, model, policy):

        self._model = model
        self._policy = policy


    def init_state(self):

        flag = False

        while flag is False:
            s_idx = random.choice(range(len(self._model._states)))
            flag = (self._model._states[s_idx]._s_index == 0)

        return s_idx, self._model._states[s_idx]

    def init_belief(self):

        b = np.zeros(len(self._model._states))

        # initialize the beliefs of the states with index=0 evenly
        num_states_each_index = pow(2, len(self._model._states[0]._prop_values))
        for i in range(num_states_each_index):
            b[i] = 1.0/num_states_each_index
            
        return b

    def observe(self, s_idx, a_idx):

        rand = np.random.random_sample()
        acc = 0.0
        for i in range(len(self._model._observations)): 
            acc += self._model._obs_fun[a_idx, s_idx, i]
            if acc > rand:
                return i, self._model._observations[i]
        else:
            sys.exit('Error in making an observation in simulation')

    def get_next_state(self, a_idx, s_idx): 

        rand = np.random.random_sample()
        acc = 0.0
        for i in range(len(self._model._states)): 
            acc += self._model._trans[a_idx, s_idx, i]
            if acc > rand:
                return i, self._model._states[i]
        else:
            sys.exit('Error in changing to the next state')

    def get_reward(self, a_idx, s_idx):

    	return self._model._reward_fun(a_idx, s_idx)

    def update(self, a_idx, o_idx, b):

        retb = np.dot(b, self._model._trans[a_idx, :])
        num_states = len(self._model._states)
        retb = [retb[i] * self._model._obs_fun[a_idx, i, o_idx] for i in range(num_states)]

        return retb/sum(retb)

    def run(self):

        [s_idx, s] = self.init_state()
        print('initial state: ' + s._name)
        b = self.init_belief()
        reward = 0

        while True:
            a_idx = self._policy.select_action(b)
            a = self._model._actions[a_idx]
            print('action selected: ' + a._name)

            reward += self.get_reward(a_idx, s_idx)

            if a._term is True: 
                break

            [s_idx, s] = self.get_next_state(a_idx, s_idx)
            print('current state: ' + s._name)

            [o_idx, o] = self.observe(s_idx, a_idx)
            print('observation made: ' + o._name)
            b = self.update(a_idx, o_idx, b)

        return reward

def main(argv):

    print('initializing model and solver')
    model = Model(0.99, ['prop2', 'prop1'], 0.9, -80.0)
    solver = Solver()

    model_name = 'model.pomdp'
    print('generating model file "' + model_name + '"')
    model.write_to_file(model_name)

    policy_name = 'output.policy'
    appl = '/home/szhang/software/pomdp_solvers/David_Hsu/appl-0.95/src/pomdpsol'
    timeout = 10
    print('computing policy "' + policy_name + '" for model "' + model_name + '"')
    print('this will take at most ' + str(timeout) + ' seconds...')
    solver.compute_policy(model_name, policy_name, appl, timeout)

    print('parsing policy: ' + policy_name)
    policy = Policy(len(model._states), len(model._actions), policy_name)

    print('starting simulation')
    simulator = Simulator(model, policy)

    num_trials = 100
    overall_reward = 0
    for i in range(num_trials): 
    	reward = simulator.run()
    	overall_reward += reward

   	print('average reward over ' + num_trials + ' is: ' + (float)overall_reward/num_trials)

if __name__ == "__main__":
    main(sys.argv)



