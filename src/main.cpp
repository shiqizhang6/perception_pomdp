
#include "Parser.h"
#include "Simulator.h"

#include <iostream>
#include <boost/numeric/ublas/io.hpp> 
#include <string>

int main(int argc, char** argv) {

    std::string path("/home/szhang/projects/2015_perception_pomdp/models/out_processed.policy");

    std::cout << "start to initialize policy parser" << std::endl; 
    Parser parser = Parser(path); 
    std::cout << "finished" << std::endl; 

    std::cout << "start to parse the policy file" << std::endl; 
    parser.parsePolicy(); 
    std::cout << "finished" << std::endl; 


    return 0; 
}
