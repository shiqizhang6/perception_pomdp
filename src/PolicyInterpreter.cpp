
#include "PolicyInterpreter.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/ublas/io.hpp>

PolicyInterpreter::PolicyInterpreter(std::string file_name) : file_name_(file_name) {

    std::cout << "working on reading policy file: " << file_name_ << std::endl; 

    if (false == boost::filesystem::exists(file_name_))
        std::cerr << "cannot find policy file: " << file_name_ << std::endl; 

    std::ifstream infile;
    infile.open(file_name_.c_str()); 

    std::string str; 

    // to get vectorLength - the number of states
    int num_states = -1; 

    while (infile >> str) {
        if (str.find("vectorLength=\"") != std::string::npos) {
            num_states = boost::lexical_cast<int> (str.substr(14, str.rfind("\"") - str.find("\"") - 1)); 
            break; 
        }
    }

    // to get numVectors
    int num_vectors = -1; 

    while (infile >> str) {
        if (str.find("numVectors=\"") != std::string::npos) {
            num_vectors = boost::lexical_cast<int> (str.substr(12, str.rfind("\"") - str.find("\"") - 1)); 
        }
    }
    
    if (-1 == num_states or -1 == num_vectors)
        std::cerr << "error in parsing policy file" << std::endl; 

    std::cout << "\tthe policy has " << num_states << " states and "
              << num_vectors << " vectors" << std::endl; 

    // create empty templates - to be filled
    policy_mat_ = boost::numeric::ublas::matrix<float> (num_vectors, num_states); 
    actions_ = boost::numeric::ublas::vector<int> (num_vectors); 

    infile.close(); 

}

void PolicyInterpreter::parsePolicy() {

    std::string str; 
    std::ifstream infile;
    infile.open(file_name_.c_str());

    while (infile >> str) {
        if (str.find("numVectors=\"") != std::string::npos) break; 
    }

    std::cout << "str: " << str << std::endl; 

    for (int i=0; i<policy_mat_.size1(); i++) {
        infile >> str; 
        infile >> str; 
        actions_[i] = boost::lexical_cast<int> (str.substr(8, str.rfind("\"") - str.find("\"") - 1)); 
        infile >> str; 
        policy_mat_(i, 0) = boost::lexical_cast<float> (str.substr(13)); 

        for (int j=1; j<policy_mat_.size2(); j++) {
            infile >> str; 
            policy_mat_(i, j) = boost::lexical_cast<float> (str); 
        }
        infile >> str; 
    }

    infile.close(); 
}

void PolicyInterpreter::selectAction(const boost::numeric::ublas::vector<float> belief, int &action) {
    
    boost::numeric::ublas::vector<float> weights;
    weights = boost::numeric::ublas::prod(belief, boost::numeric::ublas::trans(policy_mat_)); 

    double max_value = -1.0; 
    int max_index = -1; 
    for (unsigned i=0; i<weights.size(); i++) {
        max_index = (weights(i) > max_value) ? i : max_index; 
        max_value = (weights(i) > max_value) ? weights(i) : max_value; 
    }

    if (-1 == max_index) {
        std::cerr << "\terror in action selection" << std::endl; 
    }
}
