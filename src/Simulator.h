
#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <boost/numeric/ublas/matrix.hpp>
#include <vector>

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
    Simulator(int state_num, int action_num);

    std::vector<State> states;
    std::vector<Action> actions; 
    std::vector<Observation> observations; 

    State underlying_state; 
    Action action; 
    Observation observation; 
    float reward; 
    float acc_reward; 

    boost::numeric::ublas::vector<float> belief;

    void init_belief();
    void select_action(boost::numeric::ublas::vector<float> b, Action& a); 
    void update_belief(boost::numeric::ublas::vector<float>& b);
    void make_observation(State underlying_state, Observation& observation); 
};

#endif
