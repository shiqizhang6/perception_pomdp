
#include <fstream>
#include <iostream>
#include "Parser.h"

/*
    Note that here we assume the policy file has been preprocessed:

    1, first line has two numbers of "number of states" and "number of points"
    2, from the second line, first number is the action (TODO: starting from 
       0 or 1???) followed by a vector of the size of state set. 
*/

Parser::Parser(std::string filename) {

    std::ifstream input_file;
    input_file.open(filename.c_str()); 
    std::string str; 

    if (!input_file) {
        std::cout << "Error in reading the policy file" << std::endl; 
    }

    input_file >> str; 
    state_num = boost::lexical_cast<int>(str); 
    str.clear();

    input_file >> str; 
    point_num = boost::lexical_cast<int>(str);
    input_file.close(); 

    action_vector = boost::numeric::ublas::vector<int> (point_num); 
    policy_matrix = boost::numeric::ublas::matrix<float> (point_num, state_num);

    std::cout << "Policy '" << filename << "' has " 
        << boost::lexical_cast<std::string> (state_num) << " states, and " 
        << boost::lexical_cast<std::string> (point_num) << " belief points" 
        << std::endl; 

    this->filename = filename; 
}

void Parser::parsePolicy() {
    std::ifstream input_file;
    input_file.open(filename.c_str());
    std::string str; 
    input_file >> str; 
    input_file >> str; 
    for (int i=0; i<point_num; i++) {
        str.clear(); 
        input_file >> str; 
        action_vector[i] = boost::lexical_cast<int>(str); 
        str.clear(); 
        for (int j=0; j<state_num; j++) {
            input_file >> str; 
            policy_matrix(i,j) = boost::lexical_cast<float>(str);
            str.clear();
        }
    }

    input_file.close(); 
}




