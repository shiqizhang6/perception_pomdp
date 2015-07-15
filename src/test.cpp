
#include "PolicyInterpreter.h"
#include "PomdpModel.h"
#include <iostream>
#include <boost/filesystem.hpp>
#include <string>

int main(int argc, char** argv) {

    std::string model_file("/tmp/model.pomdp"), policy_file("/tmp/out.policy"); 

    PomdpModel *sim = new PomdpModel(); 
    sim->writeModelToFile(model_file); 
    
    PolicyInterpreter *policy = new PolicyInterpreter(policy_file); 
    policy->parsePolicy(); 

    // std::cout << policy->actions_ << std::endl; 
    // std::cout << policy->policy_mat_ << std::endl; 

    return 0; 
}


