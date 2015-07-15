
#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <boost/numeric/ublas/matrix.hpp>
#include "PomdpModel.h"


class Simulator {
public: 

    Simulator() {}
    Simulator(const PomdpModel * , State * &); 

    void makeObservation(PomdpModel const *, Observation & ); 
    void initBelief(boost::numeric::ublas::vector<float> & ); 
    void updateBelief(boost::numeric::ublas::vector<float> & ); 
    void updateState(State *); 
    void updateReward(float reward, float acc_reward); 

    // TODO
}; 

#endif
