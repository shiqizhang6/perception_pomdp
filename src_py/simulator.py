#! /usr/bin/env python


import random
import os
import numpy as np
import sys
from constructor import Model, State, Action, Obs
from policy import Policy, Solver

from classifier_icra2014 import ClassifierICRA
from classifier_ijcai2016 import ClassifierIJCAI

from oracle_full_ijcai2016 import TFTable


# from classifier code
import csv
import copy

class Simulator(object):

    def __init__(self, model = None, policy = None, object_prop_names = None, request_prop_names = None):
        self._model = model
        self._policy = policy
        self._object_prop_names = object_prop_names
        self._request_prop_names = request_prop_names

        self._color_values = ['brown','green','blue']
        self._weight_values = ['light','medium','heavy']
        self._content_values = ['glass','screws','beans','rice']

        self._action_cnt = 0
        self._predefined_action_sequence = [0, 1, 2, 3, 4, 6]

    def train_classifier(self):

        datapath = "../data/ijcai2016"
        behaviors = ["look","grasp","lift","hold","lower","drop","push","press"]
        modalities = ["color","hsvnorm4","vgg","shape","effort","position","audio","surf200"]

        # predicates = self._color_values + self._weight_values + self._content_values

        # minumum number of positive and negative examples needed for a predicate to be included in this experiment
        min_num_examples_per_class = 5 # in the actual experiment, this should be 4 or 5 to include more predicates; when doing a quick test, you can set it to 14 or 15 in which case only 3 predicates are valid

        # file that maps names to IDs
        objects_ids_file = "../data/ijcai2016/object_list.csv"

        # whether to load saved classifiers intead of training them
        # this should only be true if the procedure was first run once and the classifiers for all train-test splits were saved
        load_classifiers = True

        # some train parameters -- only valid if num_object_split_tests is not 32
        num_train_objects = 28
        num_trials_per_object = 5

        # whether to do internal cross validation; if false, all contexts are treated equally at test time
        # needs to be true if observation models for behaviors-predicate pairs are needed
        perform_internal_cv = True

        # how many total tests to do -- if 32, then this does object-based cross validation
        num_object_split_tests = 32

        # precompute and store train and test set ids for each test
        train_set_dict = dict()
        test_set_dict = dict()

        # create oracle
        T_oracle = TFTable(min_num_examples_per_class)
        T_oracle.loadFullAnnotations()


        # classifier = ClassifierICRA(datapath,behaviors,modalities,predicates)
        classifier = ClassifierIJCAI(datapath,behaviors,modalities,T_oracle,objects_ids_file)
        print "Classifier created..."

        # get ids
        object_ids = copy.deepcopy(classifier.getObjectIDs());

        # set them in the oracle
        classifier._T_oracle.setObjectIDs(object_ids)
        
        # filter predicates
        print("All predicates:")
        all_predicates = T_oracle.getAllPredicates()
        print(str(all_predicates))
        print("Num predicates: "+str(len(all_predicates)))

        # where to store the confusion matrices for the classification results
        pred_cm_dict = dict()
        for pred in all_predicates:
            cm_p = np.zeros( (2,2) )
            pred_cm_dict[pred]=cm_p





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
        classifier.trainClassifiers(train_object_ids,num_trials_per_object,req_train_predicates)
        
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

        if behavior == 'ask' or behavior == 'reinit':
            return self.observe_sim(s_idx, a_idx)

        target_object = '_'.join(self._object_prop_names)

        obs_distribution = np.ones((2,2,2))

        o_name = 'p'
        prob = 1.0

        for prop_name in self._request_prop_names:

            prob_list = []

            if prop_name in self._color_values:
                for v_color in self._color_values:
                    prob_list += [self._classifier.classify(target_object, behavior, v_color)]

                max_idx = prob_list.index(max(prob_list))
                o_name += str(int(prop_name == self._color_values[max_idx]))

            elif prop_name in self._weight_values:

                for v_weight in self._weight_values:
                    prob_list += [self._classifier.classify(target_object, behavior, v_weight)]

                max_idx = prob_list.index(max(prob_list))
                o_name += str(int(prop_name == self._weight_values[max_idx]))
                    
            elif prop_name in self._content_values:

                for v_content in self._color_values:
                    prob_list += [self._classifier.classify(target_object, behavior, v_content)]

                max_idx = prob_list.index(max(prob_list))
                o_name += str(int(prop_name == self._content_values[max_idx]))

            prob *= max(prob_list)

        for o_idx, o_val in enumerate(self._model._observations):
            if o_val._name == o_name:
                return o_idx, o_val, prob
        else:
            sys.exit('Error in making an observation in real')

    def observe_sim(self, s_idx, a_idx):

        rand = np.random.random_sample()
        acc = 0.0
        for i in range(len(self._model._observations)): 
            o_prob = self._model._obs_fun[a_idx, s_idx, i]
            acc += o_prob
            if acc > rand:
                return i, self._model._observations[i], o_prob
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

    def run(self, planner):

        [s_idx, s] = self.init_state()
        print('initial state: ' + s._name)
        b = self.init_belief()
        reward = 0

        while True:

            if planner == 'pomdp':
                a_idx = self._policy.select_action(b)

            elif planner == 'random':
                a_idx = random.choice(range(len(self._model._actions)))

            elif planner == 'predefined':
                # if all predefined actions have been used, take a terminal action

                if self._action_cnt == len(self._predefined_action_sequence):
                    prop_values = self._model._states[b.argmax()]._prop_values
                    fake_action = Action(True, None, prop_values)
                    for action_idx, action_val in enumerate(self._model._actions):
                        if action_val._name == fake_action._name:
                            a_idx = action_idx
                            break;
                    else:
                        sys.exit('Error in selecting predefined actions')

                else:
                    a_idx = self._predefined_action_sequence[self._action_cnt]
                    self._action_cnt += 1

            a = self._model._actions[a_idx]
            print('action selected (' + planner + '): ' + a._name)

            reward += self.get_reward(a_idx, s_idx)

            if a._term is True: 
                break

            [s_idx, s] = self.get_next_state(a_idx, s_idx)
            print('current state: ' + s._name)


            # [o_idx, o, o_prob] = self.observe_sim(s_idx, a_idx)
            [o_idx, o, o_prob] = self.observe_real(s_idx, a_idx)

            print('observation made: ' + o._name + '  probability: ' + str(o_prob))
            b = self.update(a_idx, o_idx, b)

        return reward

def main(argv):

    print('initializing model and solver')
    # a simple case that handcode the object properties
    # object_prop_names = ['heavy', 'blue', 'beans']

    num_trials = 100
    overall_reward = 0
    simulator_for_classifier = Simulator()
    classifier = simulator_for_classifier.train_classifier()

    for i in range(num_trials): 

        object_prop_names = [random.choice(simulator_for_classifier._weight_values), \
                             random.choice(simulator_for_classifier._color_values), \
                             random.choice(simulator_for_classifier._content_values)]

        print 'object: ' + str(object_prop_names)

        # the property names that human asks about
        # request_prop_names = ['heavy', 'blue']

        # 
        request_prop_names_random = [random.choice(simulator_for_classifier._weight_values), \
                                     random.choice(simulator_for_classifier._color_values), \
                                     random.choice(simulator_for_classifier._content_values)]

        request_prop_names_correct = object_prop_names

        request_prop_names = random.choice([request_prop_names_random, request_prop_names_correct])

        request_prop_names = request_prop_names[0:random.choice([1,2,3])]

        print 'request: ' + str(request_prop_names)     

        model = Model(0.99, request_prop_names, 0.85, -40.0)
        solver = Solver()

        model_name = 'model.pomdp'
        print('generating model file "' + model_name + '"')
        model.write_to_file(model_name)

        planner = 'pomdp'

        if planner == 'pomdp':

            policy_name = 'output.policy'
            appl = '/home/szhang/software/appl/appl-0.96/src/pomdpsol'
            timeout = 2
            dir_path = os.path.dirname(os.path.realpath(__file__))
            print('computing policy "' + dir_path + '/' + policy_name + '" for model "' + model_name + '"')
            print('this will take at most ' + str(timeout) + ' seconds...')
            solver.compute_policy(model_name, policy_name, appl, timeout)

            print('parsing policy: ' + policy_name)
            policy = Policy(len(model._states), len(model._actions), policy_name)

            print('starting simulation')
            simulator = Simulator(model, policy, object_prop_names, request_prop_names)

        elif planner == 'predefined' or planner == 'random':
            simulator = Simulator(model, None, object_prop_names, request_prop_names)

        else:
            sys.exit('planner selection error')

        simulator._classifier = classifier

        reward = simulator.run(planner)
        overall_reward += reward
        print 'reward: ' + str(reward) + '\n'

    print('average reward over ' + str(num_trials) + ' is: ' + str(overall_reward/float(num_trials)))

if __name__ == "__main__":
    main(sys.argv)



