
#include "Simulator.h"

std::ostream& operator<< (std::ostream& s, const Action& action) {
    if (action == COLOR) {
        s << "Color"; 
    } else if (action == CONTENT) {
        s << "Content";
    } else if (action == WEIGHT) {
        s << "Weight";
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
    for (int i=0; i < COLOR_LENGTH; i++ {
        for (int j=0; j < CONTENT_LENGTH; j++) {
            for (int k=0; k < WEIGHT_LENGTH; k++) {
                states_.push_back(new State(index++, static_cast<Color>(i), 
                    static_cast<Content>(j), static_cast<Weigth>(k)));
            }
        }
    }

    // initialize action set
    for (int i=0; i < ACTION_LENGTH; i++) {
        actions.push_back(static_cast<Action>(i));
    }

    index = 0; 
    // initialize observation set
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
    // TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
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



