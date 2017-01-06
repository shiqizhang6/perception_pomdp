#include "PomdpModel.h"

std::ostream& operator<<(std::ostream& stream, const Color& color) {

    if (color == RED) {
        stream << "red"; 
    } else if (color == GREEN) {
        stream << "green";
    } else if (color == BLUE) {
        stream << "blue"; 
    } else {
        stream << "unknown"; 
    }
    return stream; 
}

std::ostream& operator<<(std::ostream& stream, const Content& content) {
    if (content == CONTENT0) {
        stream << "content0"; 
    } else if (content == CONTENT1) {
        stream << "content1";
    } else if (content == CONTENT2) {
        stream << "content2";
    } else if (content == CONTENT3) {
        stream << "content3";
    } else {
        stream << "unknown"; 
    }
    return stream; 
}

std::ostream& operator<<(std::ostream& stream, const Weight& weight) {
    if (weight == HEAVY) {
        stream << "heavy";
    } else if (weight == MEDIUM) {
        stream << "medium";
    } else if (weight == LIGHT) {
        stream << "light";
    } else {
        stream << "unknown";
    }
    return stream; 
}

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

ObservationColor::ObservationColor(Color color) : color_(color) {

    sensing_modality_ = COLOR; 

    std::cout << color_ << std::endl; 
    std::ostringstream oss;
    oss << color_;
    name_ = oss.str(); 
}

ObservationContent::ObservationContent(Content content) : content_(content) {
    sensing_modality_ = CONTENT; 
    std::ostringstream oss; 
    oss << content_;
    name_ = oss.str(); 
}

ObservationWeight::ObservationWeight(Weight weight) : weight_(weight) {
    sensing_modality_ = WEIGHT; 
    std::ostringstream oss; 
    oss << weight_;
    name_ = oss.str(); 
}

PomdpModel::PomdpModel() {

    std::cout << "creating templates for states, action and observations" << std::endl; 

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
                actions_.push_back(new ActionTerminating(StateNonTerminal(
                    static_cast<Color>(i), static_cast<Content>(j), static_cast<Weight>(k)))); 
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
    observations_.push_back(new ObservationNone()); 


    std::string laptop_path, desktop_path, laptop_obs_path, desktop_obs_path, laptop_reward_path, desktop_reward_path; 

    laptop_path = ("/home/szhang/projects/2017_perception_pomdp/perception_pomdp/models/"); 
    desktop_path = ("/home/shiqi/projects/2015_perception_pomdp/models/"); 

    laptop_obs_path = laptop_path + "train10/"; 
    desktop_obs_path = desktop_path + "train10/"; 
    laptop_reward_path = laptop_path + "action_costs.txt"; 
    desktop_reward_path = desktop_path + "action_costs.txt"; 

    // load transition model
    std::cout << "\nworking on tra_model_ ... " << std::endl; 

    loadTraModel(); 

    // load observation model
    std::cout << "\nworking on obs_model_ ..." << std::endl; 

    if (boost::filesystem::exists(laptop_obs_path))
        loadObsModel(laptop_obs_path); 
    else if (boost::filesystem::exists(desktop_obs_path))
        loadObsModel(desktop_obs_path); 
    else
        std::cerr << "path (observation model) invalid" << std::endl; 

    // load reward model
    std::cout << "\nworking on rew_model_ ..." << std::endl; 
    if (boost::filesystem::exists(laptop_reward_path))
        loadRewModel(laptop_reward_path); 
    else if (boost::filesystem::exists(desktop_reward_path))
        loadRewModel(desktop_reward_path); 
    else
        std::cerr << "path (reward model) invalid"  << std::endl; 

    std::cout << "model initialization done" << std::endl; 
    
}

