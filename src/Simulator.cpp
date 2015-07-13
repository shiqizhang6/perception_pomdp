#include "Simulator.h"

std::ostream& operator<< (std::ostream& s, const SensingModality& sm) {
    if (sm == COLOR) {
        s << "color"; 
    } else if (sm == CONTENT) {
        s << "content";
    } else if (sm == WEIGHT) {
        s << "weight";
    } else if (sm == NONE) {
        s << "none"; 
    } else {
        s << "unknown"; 
    }
    return s; 
}

std::ostream& operator<< (std::ostream& s, const ObservationColor& o) {
    s << o;     return s; 
}
std::ostream& operator<< (std::ostream& s, const ObservationContent& o) {
    s << o;     return s; 
}
std::ostream& operator<< (std::ostream& s, const ObservationWeight& o) {
    s << o;     return s; 
}

Simulator::Simulator() {

    // initialize state set
    for (int i=0; i < COLOR_LENGTH; i++)
        for (int j=0; j < CONTENT_LENGTH; j++)
            for (int k=0; k < WEIGHT_LENGTH; k++)
                states_.push_back(new StateNonTerminalNotInHand(static_cast<Color>(i), 
                    static_cast<Content>(j), static_cast<Weight>(k)));

    for (int i=0; i < COLOR_LENGTH; i++)
        for (int j=0; j < CONTENT_LENGTH; j++)
            for (int k=0; k < WEIGHT_LENGTH; k++)
                states_.push_back(new StateNonTerminalInHand(static_cast<Color>(i), 
                    static_cast<Content>(j), static_cast<Weight>(k)));

    states_.push_back(new StateTerminal()); 

    // initialize action set
    actions_.push_back(new ActionNonTerminating(CONTENT, "drop", 0.0)); 
    actions_.push_back(new ActionNonTerminating(NONE, "grasp", 0.0)); 
    actions_.push_back(new ActionNonTerminating(WEIGHT, "lift", 0.0)); 
    actions_.push_back(new ActionNonTerminating(COLOR, "look", 0.0)); 
    actions_.push_back(new ActionNonTerminating(WEIGHT, "poke", 0.0)); 
    actions_.push_back(new ActionNonTerminating(WEIGHT, "push", 0.0)); 
    actions_.push_back(new ActionNonTerminating(CONTENT, "shake", 0.0)); 
    actions_.push_back(new ActionNonTerminating(WEIGHT, "tap", 0.0)); 

    for (int i=0; i < COLOR_LENGTH; i++) {
        for (int j=0; j < CONTENT_LENGTH; j++) {
            for (int k=0; k < WEIGHT_LENGTH; k++) {
                actions_.push_back(new ActionTerminating(
                    StateNonTerminal(static_cast<Color>(i), 
                                     static_cast<Content>(j), 
                                     static_cast<Weight>(k)))); 

            }   
        }   
    }

    // initialize observation set
    for (int i=0; i < COLOR_LENGTH; i++) {
        observations_.push_back(new ObservationColor(static_cast<Color>(i)));
    }
    for (int i=0; i < CONTENT_LENGTH; i++) {
        observations_.push_back(new ObservationContent(static_cast<Content>(i)));
    }
    for (int i=0; i < WEIGHT_LENGTH; i++) {
        observations_.push_back(new ObservationWeight(static_cast<Weight>(i)));
    }

    std::string laptop_path, desktop_path, laptop_obs_path, desktop_obs_path, laptop_reward_path, desktop_reward_path; 

    laptop_path = ("/home/szhang/projects/2015_perception_pomdp/models/"); 
    desktop_path = ("/home/shiqi/projects/2015_perception_pomdp/models/"); 

    laptop_obs_path = laptop_path + "observation_model/"; 
    desktop_obs_path = desktop_path + "observation_model/"; 
    laptop_reward_path = laptop_path + "action_costs.txt"; 
    desktop_reward_path = desktop_path + "action_costs.txt"; 

    // load transition model
    loadTraModel(); 

    // load observation model
    if (boost::filesystem::exists(laptop_obs_path))
        loadObsModel(laptop_obs_path); 
    else if (boost::filesystem::exists(desktop_obs_path))
        loadObsModel(desktop_obs_path); 
    else
        std::cerr << "path (observation model) invalid" << std::endl; 

    // load reward model
    if (boost::filesystem::exists(laptop_reward_path))
        loadRewModel(laptop_reward_path); 
    else if (boost::filesystem::exists(desktop_reward_path))
        loadRewModel(desktop_reward_path); 
    else
        std::cerr << "path (reward model) invalid"  << std::endl; 
    
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
    // if (pos == -1)
        std::cerr << "Error in action selection" << std::endl;
    // else
    //     action.index_ = parser.action_vector[pos]; 
}

