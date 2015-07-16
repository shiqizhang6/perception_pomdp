
#include "PolicyInterpreter.h"
#include "PomdpModel.h"
#include "Simulator.h"
#include <iostream>
#include <boost/filesystem.hpp>
#include <string>
#include <unistd.h>



int main(int argc, char** argv) {

    std::string model_file("/tmp/model.pomdp"), policy_file("/tmp/out.policy"); 

    PomdpModel *model = new PomdpModel(); 
    model->writeModelToFile(model_file); 
    
    PolicyInterpreter *policy = new PolicyInterpreter(policy_file); 
    policy->parsePolicy(); 

    int state_index = -1, action_index = -1, observation_index = -1; 
    float reward, acc_reward = 0; 

    std::cout << "working on simulation" << std::endl; 
    
    Simulator *sim = new Simulator(model); 
    state_index = sim->initState(); 
    std::cout << "\n\tstate: " << model->states_[state_index]->name_ << std::endl; 

    boost::numeric::ublas::vector<float> belief(sim->initBelief()); 
    std::cout << "\tbelief" << belief << std::endl; 

    // until falls into the terminal state
    while (state_index != model->states_.size() - 1) {

        policy->selectAction(belief, action_index); 
        std::cout << "\taction: " << model->actions_[action_index]->name_ << std::endl; 

        sim->makeObservation(state_index, action_index, observation_index); 
        std::cout << "\tobservation: " << model->observations_[observation_index]->name_ << std::endl; 

        sim->updateBelief(action_index, observation_index, belief); 
        std::cout << "\tbelief" << belief << std::endl; 

        sim->updateReward(action_index, state_index, reward, acc_reward); 
        std::cout << "\treward: " << reward << std::endl; 

        sim->updateState(action_index, state_index); 
        std::cout << "\n\tstate: " << model->states_[state_index]->name_ << std::endl; 

        usleep(1000000);
    }

    std::cout << "overall reward: " << acc_reward << std::endl; 

    return 0; 
}


