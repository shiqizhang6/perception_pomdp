
#include <time.h>
#include "Simulator.h"

Simulator::Simulator(const PomdpModel *model, State * &state) : model_(model) {

    srand (time(NULL));
    int state_index = rand() % (model->states_.size() - 1); 
    state = model->states_[state_index]; 
}

void Simulator::initBelief(boost::numeric::ublas::vector<float> &belief) {

    int state_num = model_->states_.size(); 

    belief = boost::numeric::ublas::vector<float>(state_num); 

    // count the number of terminal states - currently only one
    for (int i=0; i < states_.size(); i++) {
        if (true == states_[i]->is_terminal_) {
            terminal_state_num++; 
        }
    }

    for (int i=0; i < states_.size(); i++) {
        belief_[i] = (states_[i]->is_terminal_) ? (0.0) : (1.0 / (state_num - terminal_state_num)); 
    }
}


