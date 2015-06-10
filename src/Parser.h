
#ifndef PARSER_H
#define PARSER_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/lexical_cast.hpp>
#include <string>

class Parser {
public:
    Parser() {} 
    Parser(std::string filename); 

    int state_num; // the number of pomdp states
    int action_num; // the number of pomdp actions
    int point_num; // the number of points in the computed policy
    std::string filename; 

    boost::numeric::ublas::matrix<float> policy_matrix; 
    boost::numeric::ublas::vector<int> action_vector; 

    void parsePolicy(); 

}; 

#endif
