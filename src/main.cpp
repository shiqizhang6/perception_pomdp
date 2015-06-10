
#include "Parser.h"
#include "Simulator.h"

#include <iostream>
#include <boost/numeric/ublas/io.hpp> 
#include <string>

int main(int argc, char** argv) {

    std::string path("/home/szhang/projects/2015_perception_pomdp/models/out_processed.policy");

    Parser parser = Parser(path); 
    std::cout << "finished parser initialization" << std::endl; 

    std::cout << "start to parse the policy file" << std::endl; 
    parser.parsePolicy(); 
    std::cout << "finished" << std::endl; 

    Simulator sim = Simulator(73, 44, 11); 
    
    sim.initBelief(sim.belief); 

    sim.selectAction(sim.belief, sim.action, parser); 

    std::cout << "action selected: " <<
        boost::lexical_cast<std::string>(sim.action.index) << std::endl; 

    return 0; 
}
