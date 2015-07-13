
#include "Parser.h"
#include "Simulator.h"

#include <iostream>
#include <boost/filesystem.hpp>
#include <string>


int main(int argc, char** argv) {

    Simulator * sim = new Simulator(); 

    sim->writeModelToFile("/tmp/model.pomdp"); 
    
    return 0; 
}
