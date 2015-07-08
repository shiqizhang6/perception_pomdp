
#include "Parser.h"
#include "Simulator.h"

#include <iostream>
#include <boost/filesystem.hpp>
#include <string>

int main(int argc, char** argv) {

    std::string sony_path, zoidberg_path, path; 
    sony_path = "/home/szhang/projects/2015_perception_pomdp/models/out_processed.policy"; 
    zoidberg_path = "/home/shiqi/projects/2015_perception_pomdp/models/out_processed.policy"; 

    if (boost::filesystem::exists(sony_path))
        path = sony_path;
    else if (boost::filesystem::exists(zoidberg_path))
        path = zoidberg_path; 
    else
        std::cout << "Error: cannot find policy" << std::endl; 

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
