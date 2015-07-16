
#ifndef POMDPMODEL_H
#define POMDPMODEL_H

#include <boost/filesystem.hpp>
#include <boost/numeric/ublas/matrix.hpp> 
#include <boost/numeric/ublas/io.hpp> // for std::cout << matrix<..>
#include <boost/multi_array.hpp> // for high dimensional arrays
#include <boost/lexical_cast.hpp>
#include <vector>
#include "PolicyInterpreter.h"
#include "Properties.h"
#include "utilities.h"

typedef boost::multi_array<float, 3> Array3; 

enum SensingModality { COLOR, CONTENT, WEIGHT, NONE, MODALITY_LENGTH }; 


class State {
public: 

    std::string name_; 
    bool is_terminal_; 

    State() {}
    State(std::string name, bool is_terminal) : name_(name), is_terminal_(is_terminal) {}
}; 

class StateTerminal : public State {
public: 

    StateTerminal() {
        name_ = "Terminal"; 
        is_terminal_ = true; 
    }
}; 

class StateNonTerminal : public State {
public:
    Color color_;
    Content content_; 
    Weight weight_; 

    bool in_hand_; 

    StateNonTerminal(Color color, Content content, Weight weight) {

        color_ = color;
        content_ = content;
        weight_ = weight;  

        std::ostringstream stream; 
        stream << "s-" << color_ << "-" << content_ << "-" << weight_; 

        name_ = stream.str(); 
        is_terminal_ = false; 
    }
}; 

class StateNonTerminalInHand : public StateNonTerminal {
public:

    StateNonTerminalInHand(Color color, Content content, Weight weight) 
        : StateNonTerminal(color, content, weight) {
        
        name_ += "-inhand"; 
        in_hand_ = true; 
    }
}; 

class StateNonTerminalNotInHand : public StateNonTerminal {
public:

    StateNonTerminalNotInHand(Color color, Content content, Weight weight) 
        : StateNonTerminal::StateNonTerminal(color, content, weight) {
    
        name_ += "-not-inhand"; 
        in_hand_ = false; 
    }
}; 

class Action{
public:
    std::string name_; 
    bool is_terminating_; 
    float cost_; 
}; 

class ActionNonTerminating : public Action {
public: 
    SensingModality sensing_modality_; 

    ActionNonTerminating(SensingModality sensing_modality, std::string name, float cost) 
        : sensing_modality_(sensing_modality) {

        name_ = name; 
        cost_ = cost; 
        is_terminating_ = false; 
    }
}; 

class ActionTerminating : public Action {
public:
    StateNonTerminal state_non_terminal_;

    ActionTerminating(StateNonTerminal state_non_terminal) 
        : state_non_terminal_(state_non_terminal) {
        
        std::ostringstream oss; 
        oss << "a-" 
            << state_non_terminal_.color_ << "-" 
            << state_non_terminal_.content_ << "-"
            << state_non_terminal_.weight_; 
        name_ = oss.str(); 
        is_terminating_ = true; 
    }
}; 

class Observation {
public: 
    SensingModality sensing_modality_; 
    std::string name_; 
};

class ObservationColor : public Observation {
public:
    Color color_;
    ObservationColor(Color color); 
}; 

class ObservationContent : public Observation {
public:
    Content content_; 
    ObservationContent(Content content); 
}; 

class ObservationWeight : public Observation {
public:
    Weight weight_; 
    ObservationWeight(Weight weight); 
}; 

class ObservationNone : public Observation {
public:
    ObservationNone() {
        sensing_modality_ = NONE; 
        name_ = "none"; 
    }
}; 

class PomdpModel {
public:

    PomdpModel(); 

    boost::multi_array<float, 3> tra_model_; 
    boost::multi_array<float, 3> obs_model_; 
    boost::numeric::ublas::matrix<float> rew_model_; 

    void loadTraModel(); 
    void loadObsModel(const std::string );
    void loadRewModel(const std::string ); 

    void writeModelToFile(const std::string ); 

    void getStateIndices(SensingModality, int, std::vector<int> &); 
    int getTerminalStateIndex(); 
    int getNonTerminalStateIndex(Color, Content, Weight, bool);
    int getActionIndex(std::string); 
    int getObservationIndex(SensingModality, int); 

    std::vector<Action *> actions_; 
    std::vector<State *> states_;
    std::vector<Observation *> observations_; 
};

#endif /* end of SIMULATOR_H */
