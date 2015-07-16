
#include <time.h>
#include "Simulator.h"

#define RANDOM_BASE (1000)

Simulator::Simulator(const PomdpModel *model) : model_(model) {
    srand (time(NULL));
}

int Simulator::initState() {
    return rand() % (model_->states_.size() - 1); 
}

boost::numeric::ublas::vector<float> Simulator::initBelief() {

    int state_num = model_->states_.size(); 

    boost::numeric::ublas::vector<float> belief(state_num); 

    for (int i=0; i < state_num; i++) {
        belief[i] = (model_->states_[i]->is_terminal_) ? (0.0) : (1.0 / (state_num - 1)); 
    }
}

void Simulator::updateState(const int action_index, int & state_index) {
    
    float r = (rand() % RANDOM_BASE) / (1.0 * RANDOM_BASE); 
    float acc = 0; 
    bool state_updated = false; 

    for (int i=0; i<model_->states_.size(); i++) {
        acc += model_->tra_model_[action_index][state_index][i]; 
        if (r >= acc) {
            state_index = i; 
            state_updated = true; 
            return; 
        }
    }

    if (false == state_updated) {
        std::cerr << "error in updating state" << std::endl; 
    }
}

void Simulator::makeObservation(const int state_index, const int action_index, int & observation_index) {

    float r = (rand() % RANDOM_BASE) / (1.0 * RANDOM_BASE); 
    float acc = 0; 

    for (int i=0; i<model_->obs_model_.shape()[2]; i++) {
        acc += model_->obs_model_[action_index][state_index][i]; 
        if (r >= acc) {
            observation_index = i; 
            return; 
        }
    }
    std::cerr << "\terror in making observation" << std::endl; 
}

void Simulator::updateBelief(const int action_index, const int observation_index, 
        boost::numeric::ublas::vector<float> & belief) {

    boost::numeric::ublas::vector<float> new_belief(belief); 
    int state_num = model_->states_.size(); 
    float acc = 0; 

    for (int i=0; i<state_num; i++) {
        float tmp = 0; 
        for (int j=0; j<state_num; j++) {
            tmp += model_->tra_model_[action_index][j][i]; 
        }
        new_belief[i] = model_->obs_model_[action_index][i][observation_index] * tmp; 
        acc += new_belief[i]; 
    }
    belief = new_belief / acc; 
}

void Simulator::updateReward(const int action_index, const int state_index, 
        float & reward, float & acc_reward) {

    reward = model_->rew_model_(action_index, state_index); 
    acc_reward += reward; 

}
