
#ifndef FEATURESELECTOR_H
#define FEATURESELECTOR_H

#include <boost/filesystem.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <map>
#include "Properties.h"

/*
    Here we name the model files following this strategy:
    [action name] underscore [property] underscore dot txt
    Each file includes a n-by-n matrix where n is the possible values of this
    property and p(i,j) represents the probability of observing value j of this
    property given the underlying value to be i. 
*/

class FeatureSelector {
public: 
    FeatureSelector() {}; 
    FeatureSelector(std::string path_to_model); 

    void evaluateMatrixQuality(boost::numeric::ublas::matrix<float> , float &);
    std::string getEffectiveProperty(std::string); 

    void learnEffectiveProperties(); 

private:

    boost::filesystem::path path_to_model_; 
    // std::set<string> action_names_; 
    // std::set<string> property_names_; 
    std::map<std::string, std::string> property_effective_;
}; 

#endif
