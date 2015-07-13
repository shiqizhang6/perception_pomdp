
#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <boost/filesystem.hpp>
#include <boost/numeric/ublas/matrix.hpp> 
#include <boost/numeric/ublas/io.hpp> // for std::cout << matrix<..>
#include <boost/multi_array.hpp> // for high dimensional arrays
#include <vector>
#include "Parser.h"
#include "Properties.h"
#include "utilities.h"

typedef boost::multi_array<float, 3> Array3; 

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
        stream << color_ << "-" << content_ << "-" << weight_; 

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

enum SensingModality { COLOR, CONTENT, WEIGHT, NONE, MODALITY_LENGTH }; 

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
        : state_non_terminal_(state_non_terminal) {}
}; 

class Observation {
public: 
    SensingModality sensing_modality_; 
};

class ObservationColor : public Observation {
public:
    Color color_;

    ObservationColor(Color color) : color_(color) {
        sensing_modality_ = COLOR; 
    }
}; 

class ObservationContent : public Observation {
public:
    Content content_; 

    ObservationContent(Content content) : content_(content) {
        sensing_modality_ = CONTENT; 
    }
}; 

class ObservationWeight : public Observation {
public:
    Weight weight_; 

    ObservationWeight(Weight weight) : weight_(weight) {
        sensing_modality_ = WEIGHT; 
    }
}; 

class ObservationNone : public Observation {
public:
    ObservationNone() {
        sensing_modality_ = NONE; 
    }
}; 

class Simulator {
public:

    Simulator(); 

    Array3 tra_model_; 
    Array3 obs_model_; 
    boost::numeric::ublas::matrix<float> rew_model_; 

    void loadTraModel(); 
    void loadObsModel(const std::string );
    void loadRewModel(const std::string ); 

    void writeModelToFile(const std::string ); 

    void getStateIndices(SensingModality, int, std::vector<int>); 
    int getTerminalStateIndex(); 
    int getNonTerminalStateIndex(Color, Content, Weight, bool);
    int getActionIndex(std::string); 
    int getObservationIndex(SensingModality, int); 

    void initBelief(boost::numeric::ublas::vector<float> &);
    void selectAction(const boost::numeric::ublas::vector<float> &, Action &, const Parser &); 
    void updateBelief(boost::numeric::ublas::vector<float> &);
    void makeObservation(const State &, Observation &); 

    std::vector<Action *> actions_; 
    std::vector<State *> states_;
    std::vector<Observation *> observations_; 

    State state_; 
    Action action_; 
    Observation observation_; 
    boost::numeric::ublas::vector<float> belief_;

    float reward_, acc_reward_; 
};

#endif /* end of SIMULATOR_H */
