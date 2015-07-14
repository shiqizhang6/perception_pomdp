
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <string>

#include "Simulator.h"

void Simulator::writeModelToFile(std::string file) {
    
    if (boost::filesystem::exists(file)) {
        std::cout << file << " exists, moved to " << file + ".old" << std::endl; 
        boost::filesystem::rename(file, file + ".old"); 
    } 

    std::cout << "writing to: " << file << std::endl; 
    std::string str; 

    float discount_factor = DISCOUNT_FACTOR; 
    str += "\ndiscount: " + boost::lexical_cast<std::string> (discount_factor); 
    str += "\nvalues: reward\n\nstates: "; 

    for (int i=0; i<states_.size(); i++)
        str += states_[i]->name_ + " "; 

    str += "\n\nactions: "; 

    for (int i=0; i<actions_.size(); i++)
        str += actions_[i]->name_ + " "; 

    str += "\n\nobservations: "; 
    
    for (int i=0; i<observations_.size(); i++) {
        str += observations_[i]->name_ + " "; 
    }

    for (int i=0; i<tra_model_.shape()[0]; i++) {
        str += "\n\nT: " + actions_[i]->name_; 
        
        for (int j=0; j<tra_model_.shape()[1]; j++) {
            str += "\n"; 
            for (int k=0; k<tra_model_.shape()[2]; k++) {
                str += boost::lexical_cast<std::string> (tra_model_[i][j][k]) + " "; 
            }
        }
    }

    for (int i=0; i<obs_model_.shape()[0]; i++) {
        str += "\n\nO: " + actions_[i]->name_; 

        for (int j=0; j<obs_model_.shape()[1]; j++) {
            str += "\n"; 
            for (int k=0; k<obs_model_.shape()[2]; k++) {
                str += boost::lexical_cast<std::string> (obs_model_[i][j][k]) + " ";
            }
        }
    }

    str += "\n\n"; 
    for (int i=0; i<rew_model_.size1(); i++) {
        for (int j=0; j<rew_model_.size2(); j++) {
            str += "R: " + actions_[i]->name_ + "\t:\t" + states_[j]->name_; 
            str += "\t: * : *\t" + boost::lexical_cast<std::string> (rew_model_(i, j)); 
            str += "\n"; 
        }
    }

    std::ofstream outfile; 
    outfile.open(file.c_str()); 
    outfile << str; 
    outfile.close(); 
    std::cout << "done" << std::endl; 
}

