#ifndef POLICYINTERPRETER_H
#define POLICYINTERPRETER_H

#include <boost/numeric/ublas/matrix.hpp>
#include <string>
#include <iostream>
#include <fstream>

class PolicyInterpreter {
public:

    PolicyInterpreter() {} 
    PolicyInterpreter(std::string ); 

    std::string file_name_; 
    boost::numeric::ublas::matrix<float> policy_mat_; // each row corresponds to an action
    boost::numeric::ublas::vector<int> actions_; 

    void parsePolicy(); 
    void selectAction(boost::numeric::ublas::vector<float> , int ); 

}; 

#endif
