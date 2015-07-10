#include "Simulator.h"

std::ostream& operator<< (std::ostream& s, const SensingModality& sm) {
    if (sm == COLOR) {
        s << "Color"; 
    } else if (sm == CONTENT) {
        s << "Content";
    } else if (sm == WEIGHT) {
        s << "Weight";
    } else if (sm == NONE) {
        s << "None"; 
    } else {
        s << "Unknown"; 
    }
    return s; 
}

std::ostream& operator<< (std::ostream& s, const ObservationColor& o) {
    s << o; 
    return s; 
}
std::ostream& operator<< (std::ostream& s, const ObservationContent& o) {
    s << o; 
    return s; 
}
std::ostream& operator<< (std::ostream& s, const ObservationWeight& o) {
    s << o; 
    return s; 
}

Simulator::Simulator() {

    int index;

    // initialize state set
    index = 0; 
    for (int h=0; h <= 1; h++) {
        for (int i=0; i < COLOR_LENGTH; i++) {
            for (int j=0; j < CONTENT_LENGTH; j++) {
                for (int k=0; k < WEIGHT_LENGTH; k++) {
                    states_.push_back(new StateNonTerminal(index++, 
                        static_cast<Color>(i), static_cast<Content>(j), 
                        static_cast<Weigth>(k), h));
                }
            }
        }
    }
    states_.push_back(new StateTerminal(index)); 

    // initialize action set
    for (int i=0; i < ACTION_LENGTH; i++) {
        actions.push_back(static_cast<Action>(i));
    }

    // initialize observation set
    index = 0; 
    for (int i=0; i < COLOR_LENGTH; i++) {
        observations.push_back(new ObservationColor(index++, static_cast<Color>(i)));
    }
    for (int i=0; i < CONTENT_LENGTH; i++) {
        observations.push_back(new ObservationContent(index++, static_cast<Content>(i)));
    }
    for (int i=0; i < WEIGHT_LENGTH; i++) {
        observations.push_back(new ObservationWeight(index++, static_cast<Weight>(i)));
    }

}

void Simulator::initBelief(boost::numeric::ublas::vector<float> &belief) {

    int state_num = states_.size(), terminal_state_num = 0; 

    belief_ = boost::numeric::ublas::vector<float>(state_num); 

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

