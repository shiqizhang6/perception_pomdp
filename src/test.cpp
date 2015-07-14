
#include "PolicyInterpreter.h"
#include "Simulator.h"
#include <iostream>
#include <boost/filesystem.hpp>
#include <string>

int main(int argc, char** argv) {

    std::string model_file("/tmp/model.pomdp"); 

    Simulator *sim = new Simulator(); 
    sim->writeModelToFile(model_file); 
    
    PolicyInterpreter *policy = new PolicyInterpreter(model_file); 

    return 0; 
}
