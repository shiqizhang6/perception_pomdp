
#include <boost/filesystem.hpp>
#include <fstream>
#include "utilities.h"

/*  assuming states_, actions_, observations_ have been initialized
    assuming there are actions named "grasp" and "drop"     */
void Simulator::loadTraModel() {

    int action_num, state_num; 

    action_num = actions_.size();
    state_num = states_.size(); 

    tra_model_ = Array3( boost::extents[action_num][state_num][state_num] );

    Array3::index action, curr, next; 

    // initialize with all zeros
    std::fill( tra_model_.begin(), tra_model_.end(), 0.0);

    for (Array3::index a=0; a != action_num; a++)
        for (Array3::index c=0; c != state_num; c++)
            for (Array3::index n=0; n != state_num; n++)
                tra_model[a][c][n] = 0.0; 


    for (Array3::index a=0, int i=0; a != action_num; a++, i++) {
        for (Array3::index c=0, int j=0; c != state_num; c++, j++) {
            for (Array3::index n=0, int k=0; n != state_num; n++, k++) {

                // if curr is terminal OR a terminating action
                if (states_[j]->is_terminal_ or actions_[i]->is_terminating) {
                    tra_model[a][c][n] = (states_[k]->is_terminal_) ? 1.0 : 0.0; 
                    continue; 
                } 
                // otherwise, identity matrix
                tra_model[a][c][n] = (j == k) ? 1.0 : 0.0; 

                if (actions_[i]->name_.find("grasp") != std::string::npos) {
                    // if, currently, object not in hand
                    if (false == states_[j]->in_hand_) {
                        // the same object
                        if (sameProperties(states_[j], states_[k])) {
                            tra_model[a][c][n] = states_[k]->in_hand_ ? GRASP_SUCCESS_RATE : 1.0 - GRASP_SUCCESS_RATE; 
                        }
                    }
                } else if (actions_[i]->name_.find("drop") != std::string::npos) {
                    if (true == states_[j]->in_hand_) {
                        if (sameProperties(states_[j], states_[k])) {
                            tra_model[a][c][n] = states_[k]->in_hand_ ? DROP_SUCCESS_RATE : 1.0 - DROP_SUCCESS_RATE; 
                        }
                    }
                }

            }
        }
    }
}

// assuming observation probabilities separated by spaces
void Simulator::loadObsModel(const std::string path) {

    if (boost::filesystem::is_directory(path))
        std::cerr("path to observation model files does not exist"); 

    int action_num, state_num, observation_num; 

    action_num = actions_.size();
    state_num = states_.size(); 
    observation_num = observations_.size(); 

    obs_model_ = Array3( boost::extents[action_num][state_num][observation_num] );

    std::fill( obs_model_.begin(), obs_model_.end(), 0.0);

    boost::filesystem::path bpath(path); 

    for (boost::filesystem::path::iterator it = bpath.begin(); it != bpath.end(); it++) {

        // assuming file has been renamed, e.g., "grasp_color.txt"
        std::string file = path.str() + *it; 
        std::string action_name = it->substr(0, it->find("_")); 
        std::string property_name = it->substr(it->find("_") + 1, it->find(".") - it->find("_"));        
        std::cout << "working on " << file << " with action " << action_name << ", property " << property_name << std::endl; 

        Array3::index action_index = getActionIndex(action_name); 

        SensingModality sensing_modality; 
        std::ifstream infile(file); 
        std::vector<std::vector<float> > mat; 

        int a_index, s_index, o_index; 
        int modality_length; 

        if (property_name.find("color") != std::str::npos) {
            sensing_modality = COLOR; 
            modality_length = COLOR_LENGTH; 
        } else if (property_name.find("content") != std::str::npos) {
            sensing_modality = CONTENT; 
            modality_length = CONTENT_LENGTH; 
        } else if (property_name.find("weight") != std::str::npos) {
            sensing_modality = WEIGHT; 
            modality_length = WEIGHT_LENGTH; 
        } else  {
            std::cerr << "error in specify sensing modality" << std::endl; 
        }

        for (int i = 0; i < modality_length; i++) { // read the i'th row
            for (int j = 0; j < modality_length; j++) { // read the j'th colomn
                
                std::vecotr<int> state_indices; 
                getStateIndices(sensing_modality, i, state_indices); 
                int observation_index = getObservationIndex(sensing_modality, j); 

                float probability; 
                probability << infile; 

                BOOST_FOREACH(const int state_index, state_indices) {
                    obs_model_[action_index][state_index][observation_index] = probability; 
                }
            }
        }
    }
}

void Simulator::loadRewModel(const std::string file) {

    // TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 

}

void Simulator::getStateIndices(SensingModality sm, int index, std::vector<int> set) {
    
    set.clear(); 
    for (int i=0; i<states_.size(); i++) {
        if (sm == COLOR and index < COLOR_LENGTH)
            if (static_cast<Color> (index) == states_[i]->color_)
                set.push_back(i); 

        else if (sm == CONTENT and index < CONTENT_LENGTH)
            if (static_cast<Content> (index) == states_[i]->content_)
                set.push_back(i); 

        else if (sm == WEIGHT and index < WEIGHT_LENGTH)
            if (static_cast<Weight> (index) == states_[i]->weight_)
                set.push_back(i); 
    }

    if (set.empty()) {
        std::cerr << "error in getting state indices" << std::endl;
    }
}

int Simulator::getTerminalStateIndex() {
    return COLOR_LENGTH * CONTENT_LENGTH * WEIGHT_LENGTH; 
}

int Simulator::getNonTerminalStateIndex(Color color, Content content, Weight weight, bool inhand) {
    return weight 
        + (content * WEIGHT_LENGTH) 
        + (color * CONTENT_LENGTH * WEIGHT_LENGTH) 
        + (inhand * COLOR_LENGTH * CONTENT_LENGTH * WEIGHT_LENGTH); 
}

int Simulator::getActionIndex(std::string action_name) {
    for (int i=0; i<actions_.size(); i++) {
        if (actions_[i]->name_.find(name) != std::str::npos)
            return i;
    }
    std::cerr << "cannot get action index" << std::endl; 
    return -1; 
}

int Simulator::getObservationIndex(SensingModality sm, int i) {
    if (sm == COLOR and i < COLOR_LENGTH)
        return i; 
    else if (sm == CONTENT and i < CONTENT_LENGTH)
        return COLOR_LENGTH + i; 
    else if (sm == WEIGHT and i < WEIGHT_LENGTH)
        return COLOR_LENGTH + CONTENT_WEIGHT + i;
    else if (sm == NONE and i == 0)
        return COLOR_LENGTH + CONTENT_WEIGHT + WEIGHT_LENGTH; 

    std::cerr << "cannot get observation index" << std::endl; 
    return -1; 
}

