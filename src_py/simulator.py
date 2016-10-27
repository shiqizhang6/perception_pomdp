#! /usr/bin/env python


import random
import os
import numpy as np
import sys
from constructor import Model, State, Action, Obs
from policy import Policy, Solver

from classifier_icra2014 import ClassifierICRA


# from classifier code
import csv
import copy

class Simulator(object):

    def __init__(self, model, policy, object_prop_names, request_prop_names):

        self._model = model
        self._policy = policy
        self._object_prop_names = object_prop_names
        self._request_prop_names = request_prop_names
        self._classifier = self.train_classifier()

    def train_classifier(self):

        datapath = "../data/icra2014"
        behaviors = ["look","grasp","lift_slow","hold","shake","high_velocity_shake","low_drop","tap","push","poke","crush"]
        modalities = ["color","patch","proprioception","audio"]

        predicates = ['brown','green','blue','light','medium','heavy','glass','screws','beans','rice']


        classifier = ClassifierICRA(datapath,behaviors,modalities,predicates)
        print "Classifier created..."
        
        # some train parameters
        num_train_objects = 24
        num_trials_per_object = 10
        
        # how train-test splits to use when doing internal cross-validation (i.e., cross-validation on train dataset)
        num_cross_validation_tests = 5
        
        
        # get all object ids and shuffle them
        object_ids = copy.deepcopy(classifier.getObjectIDs());
        
        random.seed(1)
        random.shuffle(object_ids)
        #print object_ids

        
        # do it again to check that the random seed shuffles the same way
        #object_ids2 = classifier.getObjectIDs();
        #random.seed(1)
        #random.shuffle(object_ids2)
        #print object_ids2
        
        
        # pick random subset for train
        train_object_ids = object_ids[0:num_train_objects]
        #print train_object_ids
        print("size of train_object_ids: " + str(len(train_object_ids)))
        print("size of object_ids: " + str(len(object_ids)))
        
        # train classifier
        classifier.trainClassifiers(train_object_ids,num_trials_per_object)
        
        # perform cross validation to figure out context specific weights for each predicate (i.e., the robot should come up with a number for each sensorimotor context that encodes how good that context is for the predicate
        classifier.performCrossValidation(5)
        
        # optional: reset random seed to something specific to this evaluation run (after cross-validation it is fixed)
        random.seed(235)
        
        return classifier

    def init_state(self):


        # flag = False

        # while flag is False:
        #     s_idx = random.choice(range(len(self._model._states)))
        #     flag = (self._model._states[s_idx]._s_index == 0)

        # return s_idx, self._model._states[s_idx]

        s_name = 's0p'
        for r in self._request_prop_names:
            if r in self._object_prop_names:
                s_name += '1'
            else:
                s_name += '0'

        for s_idx, s_val in enumerate(self._model._states):
            if s_val._name == s_name:
                return s_idx, s_val
        else:
            sys.exit('Error in initializing state')


    def init_belief(self):

        b = np.zeros(len(self._model._states))

        # initialize the beliefs of the states with index=0 evenly
        num_states_each_index = pow(2, len(self._model._states[0]._prop_values))
        for i in range(num_states_each_index):
            b[i] = 1.0/num_states_each_index
            
        return b

    def observe_real(self, s_idx, a_idx):

        behavior = self._model.get_action_name(a_idx)

        target_object = '_'.join(self._object_prop_names)

        obs_distribution = np.ones((2,2,2))

        o_name = 'p'
        for prop_name in self._request_prop_names:
            prob = self._classifier.classify(target_object, behavior, prop_name)

            if prob > 0.5:
                o_name += '1'
            else:
                o_name += '0'       

        for o_idx, o_val in enumerate(self._model._observations):
            if o_val._name == o_name:
                return o_idx, o_val
        else:
            sys.exit('Error in making an observation in real')

    def observe_sim(self, s_idx, a_idx):

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

        return self._model._reward_fun[a_idx, s_idx]

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

            # [o_idx, o] = self.observe_sim(s_idx, a_idx)
            [o_idx, o] = self.observe_real(s_idx, a_idx)
            print('observation made: ' + o._name)
            b = self.update(a_idx, o_idx, b)

        return reward

def main(argv):

    print('initializing model and solver')
    # the property names of the actual on-table object
    object_prop_names = ['heavy', 'blue', 'beans']

    # the property names that human asks about
    request_prop_names = ['blue', 'glass']

    model = Model(0.99, request_prop_names, 0.85, -40.0)
    solver = Solver()

    model_name = 'model.pomdp'
    print('generating model file "' + model_name + '"')
    model.write_to_file(model_name)

    policy_name = 'output.policy'
    appl = '/home/szhang/software/pomdp_solvers/David_Hsu/appl-0.95/src/pomdpsol'
    timeout = 10
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('computing policy "' + dir_path + '/' + policy_name + 
          '" for model "' + model_name + '"')
    print('this will take at most ' + str(timeout) + ' seconds...')
    solver.compute_policy(model_name, policy_name, appl, timeout)

    print('parsing policy: ' + policy_name)
    policy = Policy(len(model._states), len(model._actions), policy_name)

    print('starting simulation')
    simulator = Simulator(model, policy, object_prop_names, request_prop_names)

    num_trials = 1
    overall_reward = 0
    for i in range(num_trials): 
        reward = simulator.run()
        overall_reward += reward

    print('average reward over ' + str(num_trials) + ' is: ' +
        str(overall_reward/float(num_trials)))

if __name__ == "__main__":
    main(sys.argv)



