
#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <boost/numeric/ublas/matrix.hpp>
#include "PomdpModel.h"


class Simulator {
public: 

    Simulator() {}
    Simulator(const PomdpModel *); 

    boost::numeric::ublas::vector<float> initBelief(); 
    int initState(); 
    void updateState(const int action_index, int & state_index); 
    void makeObservation(const int state_index, const int action_index, 
            int & observation_index); 
    void updateBelief(const int action_index, const int observation_index, 
            boost::numeric::ublas::vector<float> & ); 
    void updateReward(const int state_index, const int action_index, 
            float & reward, float & acc_reward); 

    const PomdpModel * model_; 
}; 

#endif
