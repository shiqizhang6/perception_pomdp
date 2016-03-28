
import random
import numpy as np
from constructor import Model, State, Action, Obs
from policy import Policy, Solver

class Simulator(object):

    def __init__(self, model, policy):

        self._model = model
        self._policy = policy


    def init_state(self):

        flag = False

        while flag is False:
            s_idx = random.choice(len(self._model._states))
            flag = (self._model._states[s_idx]._s_index == 0)

        return s_idx, self._model._states[s_idx]

    def init_belief(self):

        b = np.zeros(len(self._model._states))

        # initialize the beliefs of the states with index=0 evenly
        num_states_each_index = pow(2, len(self._model._states[0]._prop_values))
        for i in range(num_states_each_index):
            b = 1.0/num_states_each_index
            
        return b

    def observe(self, s_idx, a_idx):

        Obs o = None
        rand = np.random.random_sample()
        acc = 0.0
        for i in range(len(self._model._observations)): 
            acc += self._model._obs_fun[a_idx, s_idx, i]
            if acc > rand:
                return i, self._model._observations[i]
        else:
            sys.exit('Error in making an observation in simulation')

    def update(self, a_idx, o_idx):

        b = np.dot(self._belief, self._model._trans[a_idx, :])
        num_states = len(self._model._states)
        b = [b[i] * self._model._obs_fun[a_idx, i, o_idx] for i in range(num_states)]

        return b/sum(b)

    def run(self):

        [s_idx, s] = self.init_state()
        b = self.init_belief()

        while True:
            a_idx = self._policy.select_action(b)
            a = self._model._actions[a_idx]
            print('action selected: ' + a._name)

            if a._term is True: 
                break

            [o_idx, o] = self.observe(s_idx, a_idx)
            print('observation made: ' + o._name)
            b = self.update(a_idx, o_idx)

def main(argv):

    model = Model(0.99, ['red', 'can'], 0.9, -10.0)
    solver = Solver()

    model_name = 'model.pomdp'
    model.write_to_file(model_name)

    policy_name = 'output.policy'
    appl = '/home/szhang/software/pomdp_solvers/David_Hsu/appl-0.95/src/pomdpsol'
    timeout = 30
    print('computing policy "' + policy_name + '" for model "' + model_name)
    print('this will take at most ' + str(timeout) + ' seconds...')
    solver.compute_policy(model_name, policy_name, appl, timeout)

    policy = Policy(len(model._states), len(model._actions), policy_name)

    simulator = Simulator(model, policy)
    simulator.run()

if __name__ == "__main__":
    main(sys.argv)



