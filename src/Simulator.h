
#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <boost/numeric/ublas/matrix.hpp>
#include <vector>
#include "Parser.h"

struct State {
    int index;
    std::string name;  

    State() { index = -1; name = (""); }
    State(int i, std::string n) : index(i), name(n) {}
}; 

struct Action {
    int index;
    std::string name; 

    Action() { index = -1; name = (""); }
    Action(int i, std::string n) : index(i), name(n) {}
}; 

struct Observation {
    int index;
    std::string name; 

    Observation() { index = -1; name = (""); }
    Observation(int i, std::string n) : index(i), name(n) {}
};


class Simulator {
public:
    Simulator() {}
    Simulator(int state_num, int action_num, int obs_num);

    std::vector<State> states;
    std::vector<Action> actions; 
    std::vector<Observation> observations; 

    State state; 
    Action action; 
    Observation observation; 
    boost::numeric::ublas::vector<float> belief;

    float reward; 
    float acc_reward; 

    void initBelief(boost::numeric::ublas::vector<float> &;
    void selectAction(const boost::numeric::ublas::vector<float> &, Action &, const Parser &); 
    void updateBelief(boost::numeric::ublas::vector<float> &);
    void makeObservation(const State &, Observation &); 
};

#endif /* end of SIMULATOR_H */
