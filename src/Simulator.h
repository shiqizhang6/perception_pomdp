
#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <boost/numeric/ublas/matrix.hpp>
#include <vector>
#include "Parser.h"

struct State {
    int index;
    State() {index=0; }
    State(int i): index(i) {}
}; 

struct Action {
    int index;
    Action() {index=0; }
    Action(int i): index(i) {}
}; 

struct Observation {
    int index;
    Observation() {index=0; }
    Observation(int i): index(i) {}
};

class Simulator {
public:
    Simulator() {}
    Simulator(int state_num, int action_num, int obs_num);

    std::vector<State> states;
    std::vector<Action> actions; 
    std::vector<Observation> observations; 

    State underlying_state; 
    Action action; 
    Observation observation; 
    float reward; 
    float acc_reward; 

    boost::numeric::ublas::vector<float> belief;

    void initBelief(boost::numeric::ublas::vector<float> &belief);
    void selectAction(const boost::numeric::ublas::vector<float> &b, Action &a, 
        const Parser& parser); 
    void updateBelief(boost::numeric::ublas::vector<float> &b);
    void makeObservation(const State &underlying_state, Observation &observation); 
};

#endif
