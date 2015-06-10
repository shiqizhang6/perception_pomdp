
#include "Simulator.h"

Simulator::Simulator(int state_num, int action_num, int obs_num) {
    for (int i=0; i<state_num; i++) {
        State s = State(i); 
        states.push_back(s); 
    }
    for (int i=0; i<action_num; i++) {
        Action a = Action(i);
        actions.push_back(a);
    }
    for (int i=0; i<obs_num; i++) {
        Observation o = Observation(i);
        observations.push_back(o);
    }
}

void Simulator::initBelief(boost::numeric::ublas::vector<float> &belief) {
    int state_num = states.size(); 
    belief = boost::numeric::ublas::vector<float>(state_num); 
    for (int i=0; i<state_num; i++) {
        belief[i] = (i==state_num-1) ? 0.0 : 1.0/(state_num-1.0); 
    }
}


void Simulator::selectAction(const boost::numeric::ublas::vector<float> &b, 
    Action &action, const Parser &parser) {

    boost::numeric::ublas::vector<float> vec;
    vec = boost::numeric::ublas::prod(parser.policy_matrix, b);

    int argmax = INT_MIN, pos=-1; 
    for (int i=0; i<vec.size(); i++) {
        pos = (vec[i] > argmax) ? i : pos;
        argmax = (vec[i] > argmax) ? vec[i] : argmax; 
    }
    if (pos == -1)
        std::cout << "Error in action selection" << std::endl;
    else
        action.index = parser.action_vector[pos]; 
}



