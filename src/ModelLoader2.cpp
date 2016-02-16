
#include <boost/filesystem.hpp>
#include <fstream>
#include <map>
#include "PomdpModel.h"

/*  assuming states_, actions_, observations_ have been initialized
    assuming there are actions named "grasp" and "drop"     */
void PomdpModel::loadTraModel() {

    int action_num, state_num; 

    action_num = actions_.size();
    state_num = states_.size(); 

    // note that boost::reshape won't work coz the # of elements grow
    Array3::extent_gen extents;
    tra_model_.resize(extents[action_num][state_num][state_num]);

    Array3::index action, curr, next; 

    std::cout << "\tinitialize tra_model_ with all zeros" << std::endl; 

    for (Array3::index a=0; a != action_num; a++)
        for (Array3::index c=0; c != state_num; c++)
            for (Array3::index n=0; n != state_num; n++)
                tra_model_[a][c][n] = 0.0; 

    Array3::index a, c, n;
    int i, j, k; 
    for (a=0, i=0; a != action_num; a++, i++) {
        for (c=0, j=0; c != state_num; c++, j++) {
            for (n=0, k=0; n != state_num; n++, k++) {

                // if curr is terminal OR a terminating action
                if (states_[j]->is_terminal_ or actions_[i]->is_terminating_) {
                    tra_model_[a][c][n] = (states_[k]->is_terminal_) ? 1.0 : 0.0; 
                    continue; 
                } 
                // otherwise, identity matrix
                tra_model_[a][c][n] = (j == k) ? 1.0 : 0.0; 

                if (actions_[i]->name_.find("grasp") != std::string::npos) {
                    // if, currently, object not in hand
                    if (false == static_cast<StateNonTerminal*> (states_[j])->in_hand_) {

                        // the same object
                        StateNonTerminal *pt1 = static_cast<StateNonTerminal*> (states_[j]), 
                                         *pt2 = static_cast<StateNonTerminal*> (states_[k]); 

                        if (pt1->color_ == pt2->color_ and pt1->content_ == pt2->content_ 
                            and pt1->weight_ == pt2->weight_) {

                            tra_model_[a][c][n] = static_cast<StateNonTerminal*> (states_[k])->in_hand_ 
                                ? GRASP_SUCCESS_RATE : 1.0 - GRASP_SUCCESS_RATE; 
                        }
                    }
                } else if (actions_[i]->name_.find("drop") != std::string::npos) {
                    if (true == static_cast<StateNonTerminal*> (states_[j])->in_hand_) {

                        StateNonTerminal *pt1 = static_cast<StateNonTerminal*> (states_[j]), 
                                         *pt2 = static_cast<StateNonTerminal*> (states_[k]); 

                        if (pt1->color_ == pt2->color_ and pt1->content_ == pt2->content_ 
                            and pt1->weight_ == pt2->weight_) {

                            tra_model_[a][c][n] = static_cast<StateNonTerminal*> (states_[k])->in_hand_ 
                                ? 1.0 - DROP_SUCCESS_RATE : DROP_SUCCESS_RATE; 
                        }
                    }
                }
            }
        }
    }
    std::cout << "\tfinished tra_model_ initialization" << std::endl; 
}

// assuming observation probabilities separated by spaces
void PomdpModel::loadObsModel(const std::string path) {

    if (false == boost::filesystem::is_directory(path))
        std::cerr << "path does not exist: " << path << std::endl; 

    int action_num, state_num, observation_num; 

    action_num = actions_.size();
    state_num = states_.size(); 
    observation_num = observations_.size(); 

    // note that boost::reshape won't work here coz the # of elements grows from (0,0,0)
    Array3::extent_gen extents;
    obs_model_.resize(extents[action_num][state_num][observation_num]);

    // initialize with all zeros
    Array3::index a;
    for (a=0; a != action_num; a++) {

        Array3::index c; 
        for (c=0; c != state_num; c++) {
        
            Array3::index n; 
            for (n=0; n != observation_num; n++) {
                if ( (actions_[a]->is_terminating_ or states_[c]->is_terminal_) 
                        and observations_[n]->sensing_modality_ == NONE)

                    obs_model_[a][c][n] = 1.0; 
                else
                    obs_model_[a][c][n] = 0.0; 
            }
        }
    }

    boost::filesystem::path bpath(path); 
    boost::filesystem::directory_iterator it(bpath), end_it; 

    std::vector<int> state_indices; 

    // learn the mapping from an action to a property
    FeatureSelector feature_selector(path); 
    feature_selector.learnEffectiveProperties(); 

    // for each file such as "look_color.txt"
    for ( ; it != end_it; it++) {

        std::string filename = it->path().filename().string(); 

        // assuming file has been renamed in a way such as "look_color.txt"
        std::cout << "\tfile name: " << filename << " " << std::endl;

        std::string file = path + filename; 

        // get the names of action and property
        std::string action_name = filename.substr(0, filename.find("_")); 
        std::string property_name = filename.substr(filename.find("_") + 1, 
            filename.rfind("_") - filename.find("_") - 1);

        std::string effective_property = feature_selector.getEffectiveProperty(action_name); 
        if (effective_property.compare(property_name) != 0)
            continue; 

        Array3::index action_index = getActionIndex(action_name); 

        SensingModality sensing_modality; 
        std::ifstream infile; 
        infile.open(file.c_str()); 

        int modality_length; 

        if (property_name.find("color") != std::string::npos) {
            sensing_modality = COLOR; 
            modality_length = COLOR_LENGTH; 
        } else if (property_name.find("content") != std::string::npos) {
            sensing_modality = CONTENT; 
            modality_length = CONTENT_LENGTH; 
        } else if (property_name.find("weight") != std::string::npos) {
            sensing_modality = WEIGHT; 
            modality_length = WEIGHT_LENGTH; 
        } else  {
            std::cerr << "error in specify sensing modality" << std::endl; 
        }

        for (int i = 0; i < modality_length; i++) { // read the i'th row
            for (int j = 0; j < modality_length; j++) { // read the j'th colomn
                
                state_indices.clear(); 
                getStateIndices(sensing_modality, i, state_indices); 
                int observation_index = getObservationIndex(sensing_modality, j); 

                float probability; 
                std::string tmp; 
                infile >> tmp; 
                probability = boost::lexical_cast<float>(tmp); 

                // terminal state does not apply
                for (int k=0; k<state_indices.size(); k++) {
                    obs_model_[action_index][state_indices[k]][observation_index] = probability; 
                }
            }
        }

        infile.close(); 
    }

    // action grasp senses nothing
    Array3::index action_grasp_index = getActionIndex("grasp");
    int observation_index = getObservationIndex(NONE, 0); 

    for (int i=0; i<state_num-1; i++) {
        obs_model_[action_grasp_index][i][observation_index] = 1.0; 
    }

    // actions drop, shake and lift sense nothing, when object is not in hand
    for (Array3::index a=0; a != action_num; a++) {

        if (actions_[a]->name_.find("drop") != std::string::npos
                or actions_[a]->name_.find("shake") != std::string::npos
                or actions_[a]->name_.find("lift") != std::string::npos) {

            for (Array3::index c=0; c != state_num; c++) {
                
                if (states_[c]->is_terminal_)
                    continue; 
                
                if (false == (static_cast<StateNonTerminal *> (states_[c]))->in_hand_ ) {

                    for (Array3::index n=0; n != observation_num; n++)
                        obs_model_[a][c][n] = (observations_[n]->sensing_modality_ == NONE) ? 1.0 : 0.0; 
                }
            }
        }
    }

    // actions poke, push and tap sense nothing, when object is in hand
    for (Array3::index a=0; a != action_num; a++) {

        if (actions_[a]->name_.find("poke") != std::string::npos
                or actions_[a]->name_.find("push") != std::string::npos
                or actions_[a]->name_.find("tap") != std::string::npos) {

            for (Array3::index c=0; c != state_num; c++) {
                
                if (states_[c]->is_terminal_)
                    continue; 

                if (true == (static_cast<StateNonTerminal *> (states_[c]))->in_hand_ ) { 
                    for (Array3::index n=0; n != observation_num; n++)
                        obs_model_[a][c][n] = (observations_[n]->sensing_modality_ == NONE) ? 1.0 : 0.0; 
                }
            }
        }
    }
}

void PomdpModel::loadRewModel(const std::string file) {

    if (boost::filesystem::exists(file))
        std::cout << "reading action-cost file: " << file << std::endl; 
    else 
        std::cerr << "path to reward file does not exist" << std::endl; 

 
    std::ifstream infile; 
    infile.open(file.c_str()); 

    std::map<std::string, float> cost_map; 
    std::string action_name; 
    float action_cost; 

    while (infile >> action_name) {
        infile >> action_cost; 
        cost_map[action_name] = action_cost; 
        std::cout << "\t" << action_name << ": " << action_cost << std::endl; 
    }

    int action_num, state_num; 

    action_num = actions_.size();
    state_num = states_.size(); 

    rew_model_ = boost::numeric::ublas::matrix<float> (action_num, state_num); 

    Color a_color; 
    Content a_content; 
    Weight a_weight; 

    for (int i=0; i<actions_.size(); i++) {
        // std::cout << "action name: " << actions_[i]->name_ << std::endl; 
        for (int j=0; j<states_.size(); j++) {

            if (true == states_[j]->is_terminal_) {

                rew_model_(i,j) = 0.0; 

            } else if (true == actions_[i]->is_terminating_) {

                a_color = static_cast<ActionTerminating *> (actions_[i])->state_non_terminal_.color_; 
                a_content = static_cast<ActionTerminating *> (actions_[i])->state_non_terminal_.content_;
                a_weight = static_cast<ActionTerminating *> (actions_[i])->state_non_terminal_.weight_;

                if (a_color == static_cast<StateNonTerminal *>(states_[j])->color_ and 
                    a_content == static_cast<StateNonTerminal *>(states_[j])->content_ and 
                    a_weight == static_cast<StateNonTerminal *>(states_[j])->weight_) {

                    rew_model_(i, j) = SUCCESS_REWARD; 

                } else {

                    rew_model_(i, j) = FAILURE_PENALTY; 

                }

            } else if (cost_map.find(actions_[i]->name_) != cost_map.end()) {
                
                rew_model_(i, j) = (-1.0) * cost_map[actions_[i]->name_]; 
                
            } else {

                std::cerr << "unknown cost of action: " << actions_[i]->name_ << std::endl; 

            }
        }
    }
    infile.close(); 
}

void PomdpModel::getStateIndices(SensingModality sm, int index, std::vector<int> &set) {
    
    set.clear(); 

    for (int i=0; i<states_.size() - 1; i++) {

        if (sm == COLOR and index < COLOR_LENGTH) {
            //std::cout << "sm == COLOR" << std::endl; 
            if (index == static_cast<StateNonTerminal *>(states_[i])->color_) {
                //std::cout << "push_back: " << i << std::endl; 
                set.push_back(i); 

            }
        } else if (sm == CONTENT and index < CONTENT_LENGTH) {
            //std::cout << "sm == CONTENT" << std::endl; 
            if (index == static_cast<StateNonTerminal *>(states_[i])->content_) {
                //std::cout << "push_back: " << i << std::endl; 
                set.push_back(i); 
            }
        } else if (sm == WEIGHT and index < WEIGHT_LENGTH) {
            //std::cout << "sm == WEIGHT" << std::endl; 
            if (index == static_cast<StateNonTerminal *>(states_[i])->weight_) {
                //std::cout << "push_back: " << i << std::endl; 
                set.push_back(i); 
            }
        } 
    }
}

int PomdpModel::getTerminalStateIndex() {
    return COLOR_LENGTH * CONTENT_LENGTH * WEIGHT_LENGTH; 
}

int PomdpModel::getNonTerminalStateIndex(Color color, Content content, Weight weight, bool inhand) {
    return weight 
        + (content * WEIGHT_LENGTH) 
        + (color * CONTENT_LENGTH * WEIGHT_LENGTH) 
        + (inhand * COLOR_LENGTH * CONTENT_LENGTH * WEIGHT_LENGTH); 
}

int PomdpModel::getActionIndex(std::string action_name) {
    for (int i=0; i<actions_.size(); i++) {
        if (actions_[i]->name_.find(action_name) != std::string::npos)
            return i;
    }
    std::cerr << "cannot get action index" << std::endl; 
    return -1; 
}

int PomdpModel::getObservationIndex(SensingModality sm, int i) {
    if (sm == COLOR and i < COLOR_LENGTH)
        return i; 
    else if (sm == CONTENT and i < CONTENT_LENGTH)
        return COLOR_LENGTH + i; 
    else if (sm == WEIGHT and i < WEIGHT_LENGTH)
        return COLOR_LENGTH + CONTENT_LENGTH + i;
    else if (sm == NONE and i == 0)
        return COLOR_LENGTH + CONTENT_LENGTH + WEIGHT_LENGTH; 

    std::cerr << "cannot get observation index" << std::endl; 
    return -1; 
}

