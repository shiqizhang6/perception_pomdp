
#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <boost/numeric/ublas/matrix.hpp>
#include <vector>
#include "Parser.h"
#include "Properties.h"

class State {
protected:
    int index_;
    bool is_terminal_; 
}

class StateTerminal : State {
    std::string name_; 

    StateTerminal() {}

    StateTerminal(int index) : index_(index) {
        name_ = "Terminal"; 
        is_terminal_ = true; 
    }
}

class StateNonTerminal : State {
    Color color_;
    Content content_; 
    Weight weight_; 

    std::string name_;  

    StateNonTerminal() {}

    StateNonTerminal(int index, Color color, Content content, Weight weight) 
        : index_(index), color_(color), content_(content), weight_(weight) {

        std::ostringstream stream; 
        stream << color_ << "-" << content_ << "-" << weight_; 
        name_ = stream.str(); 
        is_terminal_ = false; 
    }
}; 

enum Action { COLOR, CONTENT, WEIGHT, ACTION_LENGTH }; 

class Observation {
protected: 
    int index_;
};

class ObservationColor : Observation {
    Color color_;
    ObservationColor(int index, Color color) : index_(index), color_(color) {}
}; 

class ObservationContent : Observation {
    Content content_; 
    ObservationContent(int index, Content content) : index_(index), content_(content) {}
}; 

class ObservationWeight : Observation {
    Weight weight_; 
    ObservationWeight(int index, Weight weight) : index_(index), weight_(weight) {}
}

class Simulator {

public:
    Simulator() {}

    void initBelief(boost::numeric::ublas::vector<float> &;
    void selectAction(const boost::numeric::ublas::vector<float> &, Action &, const Parser &); 
    void updateBelief(boost::numeric::ublas::vector<float> &);
    void makeObservation(const State &, Observation &); 

private:
    std::vector<State *> states_;
    std::vector<Action> actions_; 
    std::vector<Observation *> observations_; 

    State state_; 
    Action action_; 
    Observation observation_; 
    boost::numeric::ublas::vector<float> belief_;

    float reward_, acc_reward_; 
};

#endif /* end of SIMULATOR_H */
