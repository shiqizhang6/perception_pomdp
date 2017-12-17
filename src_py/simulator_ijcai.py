#! /usr/bin/env python


import random
import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from constructor import Model, State, Action, Obs
from policy import Policy, Solver

from classifier_ijcai2016 import ClassifierIJCAI


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

        # with "ask" action
        self._predefined_action_sequence = [8, 0, 7, 1, 2, 3, 4, 5]
        self._legal_actions = {
            0: [0,8], 
            1: [1,6,7], 
            2: [2], 
            3: [3,4], 
            4: [5], 
            5: [9]
        }

        # without "ask" action
        # self._predefined_action_sequence = [0, 7, 1, 2, 3, 4, 5]

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

    def observe_real_ijcai(self, s_idx, a_idx, request_prop_names, test_object_index): 

        #print('Making an observation... ')
        query_behavior_in_list = self._model.get_action_name(a_idx)

        if query_behavior_in_list == 'ask':
            return self.observe_sim(s_idx, a_idx)

        query_trial_index = random.randrange(1, 5)

        # print('test_object_index: ' + str(test_object_index))
        # print('query_behavior_in_list: ' + str(query_behavior_in_list))
        #print('request_prop_names: ' + str(request_prop_names))
        #print('query_trial_index: ' + str(query_trial_index))

        list_with_single_behavior = []
        list_with_single_behavior.append(query_behavior_in_list)

        query_pred_probs = self._model._classifiers[test_object_index].classifyMultiplePredicates(\
            test_object_index, list_with_single_behavior, request_prop_names, query_trial_index)

        print("\nObservation predicates and probabilities:")
        print(request_prop_names)
        # print(query_pred_probs)

        obs_name = 'p'
        for prob in query_pred_probs:
            if prob > 0.5: 
                obs_name += '1'
            else:
                obs_name += '0'

        for o_idx, o_val in enumerate(self._model._observations): 
            if o_val._name == obs_name:
                return [o_idx, o_val, 0.0]
        else:
            print 'Error in observe_real_ijcai'


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

    # according to the current belief distribution, the robot is forced to select a
    # report action to terminate the exploration process
    # it is used in random_plus strategy. 
    def select_report_action(self, b):

    	# it's possible that the most likely entry goes to the term state -- we need to 
    	# assign zero to the term state to make sure we find an report action that makes sense
    	b_non_term = b
    	b_non_term[-1] = 0.0

        prop_values = self._model._states[b_non_term.argmax()]._prop_values
        fake_action = Action(True, None, prop_values)
        for action_idx, action_val in enumerate(self._model._actions):
            if action_val._name == fake_action._name:
                a_idx = action_idx
                break;
        else:
            sys.exit('Error in selecting predefined actions')

        return a_idx


    def run(self, planner, request_prop_names, test_object_index, max_cost):

        [s_idx, s] = self.init_state()
        print('initial state: ' + s._name)
        b = self.init_belief()
        trial_reward = 0
        action_cost = 0
        self._action_cnt = 0

        while True:

            # select an action using the POMDP policy
            if planner == 'pomdp':
                a_idx = self._policy.select_action(b)


            # this a weakest policy, an action is randomly selected from the exploration and
            # claiming actions. Illegal actions lead to early termination
            elif planner == 'random':
                a_idx = random.choice(range(len(self._model._actions)))


            # exploration actions are randomly selected from the legal actions for each
            # state. E.g., in state x_1, the robot can only press or push the obj
            # we set a cost threshold - once it's reached the robot selects the most 
            # likely claim to terminate the exploration
            elif planner == 'random_plus':
                
                # the robot takes legal actions until their total cost goes beyond the max_cost
                # this trial_reward does not include the cost of the report action. 
                print ("abs(trial_reward)")
                print(abs(trial_reward))
                if abs(trial_reward) > max_cost:
                    a_idx = self.select_report_action(b)
                else: 
                    a_idx = random.choice(self._legal_actions[self._model._states[s_idx]._s_index])
                    

            # the robot strictly follows this predefined sequence of actions. The robot 
            # may fail in some exploration actions. These failures are ignored - as a result
            # the robot may take illegal actions resulting in undesirable early terminations
            elif planner == 'predefined':
                # if all predefined actions have been used, take a terminal action

                if self._action_cnt < len(self._predefined_action_sequence):

                    a_idx = self._predefined_action_sequence[self._action_cnt]
                    self._action_cnt += 1

                else: 
                    a_idx = self.select_report_action(b)


            # improved predefined strategy, the robot follows the predefined sequence
            # of actions but only moves to the next action when it is legal. 
            elif planner == 'predefined_plus':

                if self._action_cnt < len(self._predefined_action_sequence):

                    a_idx = self._predefined_action_sequence[self._action_cnt]
                    print ('a_idx: ' + str(a_idx))
                    print ('s_idx: ' + str(s_idx))
                    print ('self._model._states[s_idx]._s_index: ' + str(self._model._states[s_idx]._s_index))

                    if a_idx in self._legal_actions[self._model._states[s_idx]._s_index]:

                        self._action_cnt += 1
                        print ('self._action_cnt: ' + str(self._action_cnt))

                    else:

                    	print('illegal')
                        a_idx = self._predefined_action_sequence[self._action_cnt - 1]
                        print ('illegal: a_idx: ' + str(a_idx))
                        assert a_idx in self._legal_actions[self._model._states[s_idx]._s_index]

                else: 
                    a_idx = self.select_report_action(b)

            else:
                sys.exit('planner type unrecognized: ' + planner)


            a = self._model._actions[a_idx]
            print('action selected (' + planner + '): ' + a._name)

            # computing reward: current state and selected action
            reward = self.get_reward(a_idx, s_idx)
            trial_reward += reward
            

            # state transition
            [s_idx, s] = self.get_next_state(a_idx, s_idx)
            print('resulting state: ' + s._name)

            # compute accumulated reward
            if s._term is True: 
                action_cost = trial_reward - reward
                break

            # make observation
            # if an action (look, grasp, etc) is unsuccessful, one will end up with no state change and a 'na' observation
            if s._name == 'terminal' or 's5': 
                [o_idx, o, o_prob] = self.observe_sim(s_idx, a_idx)
                # [o_idx, o, o_prob] = self.observe_real(s_idx, a_idx)
            else: 
                [o_idx, o, o_prob] = self.observe_real_ijcai(s_idx, a_idx, request_prop_names, test_object_index)

            print('observation made: ' + o._name + '  probability: ' + str(o_prob))

            # update belief
            b = self.update(a_idx, o_idx, b)
            print("Belief: " + str(["%0.2f" % i for i in b]))

        return trial_reward, action_cost

def main(argv):

    print('initializing model and solver')

    num_trials = 200

    predicates = ['text', 'yellow', 'bright', 'half-full', 'silver', 'rattles', \
    'aluminum', 'large', 'small', 'round', 'heavy', 'container', 'tube', 'red', \
    'can', 'full', 'water', 'narrow', 'hollow', 'top', 'plastic', 'white', 'empty', \
    'wide', 'cap', 'cylinder', 'lid', 'metallic', 'circular', 'canister', 'medium-sized', \
    'tall', 'short', 'liquid', 'light', 'metal', 'bottle']

    printout = ''
    
    df=pd.DataFrame()                                                 #Creating a dataframe for plotting

    for planner in ['pomdp', 'predefined', 'predefined_plus', 'random', 'random_plus']:
    #for planner in ['pomdp','random_plus', 'predefined_plus']:

        for num_props in [1, 2, 3]: 

            overall_reward = 0
            overall_action_cost = 0
            success_trials = 0
            max_cost = 50

            for i in range(num_trials): 

                print("\n##################### starting a new trial ######################\n")

                print('Trial: ' + str(i) + '/' + str(num_trials-1))

                # the user can ask about at most 3 predicates
                query_length = random.randrange(num_props, num_props+1)
                request_prop_names = random.sample(predicates, query_length)

                # request_prop_names = ['bright', 'half-full']

                # we use totally 32 objects, after filtering out the ones with little training data
                test_object_index = random.randrange(1, 33)

                print('request_prop_names: ' + str(request_prop_names))
                model = Model(0.99, request_prop_names, 0.7, -50.0, test_object_index)
                model.write_to_file('model.pomdp')

                print 'Predicates: ', request_prop_names
                # for p in request_prop_names: 
                #     print(model._classifiers[test_object_index].isPredicateTrue(p, str(test_object_index)))

                object_prop_names = []
                for p in predicates:
                    if model._classifiers[test_object_index].isPredicateTrue(p, str(test_object_index)):
                        object_prop_names.append(p)
                # print 'object_prop_names: ', object_prop_names

                # request_prop_names_random = [random.choice(simulator_for_classifier._weight_values), \
                #                            random.choice(simulator_for_classifier._color_values), \
                #                            random.choice(simulator_for_classifier._content_values)]

                solver = Solver()

                model_name = 'model.pomdp'
                print('generating model file "' + model_name + '"')
                model.write_to_file(model_name)

                # planner = 'pomdp'
                # look -> press -> grasp -> lift -> hold -> lower -> drop 
                # planner = 'predefined'

                if planner == 'pomdp':

                    policy_name = 'output.policy'
                    applPath1 = '/home/szhang/software/appl/appl-0.96/src/pomdpsol'
                    applPath2 = '/home/szhang/software/pomdp_solvers/David_Hsu/appl-0.95/src/pomdpsol'
                    applPath3 = '/home/saeid/software/sarsop/src/pomdpsol' 
                    pathlist=[applPath1,applPath2,applPath3]
                    appl=None
                    for p in pathlist:
                        if os.path.exists(p):
                            appl=p   
                    if appl==None:
                        print "ERROR: No path detected for pomdpsol"


                    timeout = 5
                    dir_path = os.path.dirname(os.path.realpath(__file__))
                    print('computing policy "' + dir_path + '/' + policy_name + '" for model "' + model_name + '"')
                    print('this will take at most ' + str(timeout) + ' seconds...')
                    solver.compute_policy(model_name, policy_name, appl, timeout)

                    print('parsing policy: ' + policy_name)
                    policy = Policy(len(model._states), len(model._actions), policy_name)

                    print('starting simulation')
                    simulator = Simulator(model, policy, object_prop_names, request_prop_names)


                elif planner == 'random' or planner == 'random_plus' or planner == 'predefined' or planner == 'predefined_plus':

                    simulator = Simulator(model, None, object_prop_names, request_prop_names)

                else:
                    sys.exit('planner selection error')


                trial_reward, action_cost = simulator.run(planner, request_prop_names,test_object_index, max_cost)

                overall_reward += trial_reward
                overall_action_cost += action_cost
                print 'overall action cost: ' + str(action_cost)
                print 'overall reward: ' + str(trial_reward) + '\n'

                success_trials += trial_reward - action_cost > 0

            printout += '\n' + planner + str(num_props) + '\n'
            printout += ('average reward over ' + str(num_trials) + ' is: ' + str(overall_reward/float(num_trials)))  + '\n'
            printout += ('average action cost over '+ str(num_trials) + ' is: ' + str(overall_action_cost/float(num_trials)))  + '\n'
            printout += ('success rate: ' + str(float(success_trials)/num_trials))  + '\n'
            print printout
            
            #Storing the results in a pandas.DataFrame for plotting
            df.at[planner+ str(num_props),'Overall reward']= overall_reward/float(num_trials)
            df.at[planner+ str(num_props),'Overall cost']= overall_action_cost/float(num_trials)
            df.at[planner+ str(num_props),'success rate']= float(success_trials)/num_trials           

    print printout
    print df

    fig=plt.figure()
    
    #Creating plots for different planners and three predicates
    for count,metric in enumerate(list(df)):
        ax=plt.subplot(1,len(list(df)),count+1)

        l1 = plt.plot([1,2,3],df.loc['pomdp1':'pomdp3',metric],marker='*',linestyle='-',label='Pomdp')
        l2 = plt.plot([1,2,3],df.loc['random_plus1':'random_plus3',metric],marker='x',linestyle='-.',label='random_plus')
        l3 = plt.plot([1,2,3],df.loc['predefined_plus1':'predefined_plus3',metric],marker='o',linestyle='--',label='predefined_plus')
        l4 = plt.plot([1,2,3],df.loc['predefined1':'predefined3',metric],marker='D',linestyle=':',label='predefined')
        l5 = plt.plot([1,2,3],df.loc['random':'random3',metric],marker='^',linestyle='-.',label='random')
        plt.ylabel(metric)
        plt.xlabel('Number of Properties')
        plt.xlim(0,3.5 )
    
    ax.legend(loc='upper left', bbox_to_anchor=(-2.0, -0.05),  shadow=True, ncol=5)
    
    plt.show()
    fig.savefig('Results_200 trials')



if __name__ == "__main__":
    main(sys.argv)



