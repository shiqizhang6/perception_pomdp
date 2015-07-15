
#include <time.h>
#include "Simulator.h"

Simulator::Simulator(PomdpModel const * model, State * &state) {

    srand (time(NULL));
    int state_index = rand() % (model->states_.size() - 1); 
    state = model->states_[state_index]; 
}

void Simulator::initBelief(boost::numeric::ublas::vector<float> &belief) {

}


