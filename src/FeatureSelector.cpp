
#include "FeatureSelector.h"

FeatureSelector::FeatureSelector (std::string path_to_model) {

    path_to_model_ = boost::filesystem::path(path_to_model); 

    std::cout << "directory " << path_to_model_; 
    if (boost::filesystem::is_directory(path_to_model_) == false) {
        std::cout << " does not exist" << std::endl; 
    } 
    else {
        std::cout << ": parsing observation model" << std::endl;
    }
}

void FeatureSelector::evaluateMatrixQuality(boost::numeric::ublas::matrix<float> mat, float &value) {

    std::cout << mat << std::endl; 

    value = 1.0; 
    for (unsigned i=0; i<mat.size1(); i++) {
        float entropy = 0.0; 
        for (unsigned j=0; j<mat.size2(); j++) {
            if (mat(i,j) != 0.0)
                entropy -= mat(i,j) * log(mat(i,j)); 
        }
        value *= entropy; 
    }
    std::cout << value << std::endl; 
}

void FeatureSelector::learnEffectiveProperties() {

    boost::filesystem::directory_iterator it(path_to_model_), end_it; 

    // [action, property] --> quality value
    std::map<std::string, float> quality_of_action; 

    for ( ; it != end_it; it++) {
        std::string file_name = it->path().filename().string(); 
        std::string action_name = file_name.substr(0, file_name.find("_")); 
        std::string property_name = file_name.substr(file_name.find("_") + 1, 
            file_name.rfind("_") - file_name.find("_") - 1); 

        std::ifstream infile; 
        infile.open((path_to_model_.string() + file_name).c_str()); 

        int modality_length; 

        if (property_name.find("color") != std::string::npos) {
            modality_length = COLOR_LENGTH; 
        } else if (property_name.find("content") != std::string::npos) {
            modality_length = CONTENT_LENGTH; 
        } else if (property_name.find("weight") != std::string::npos) {
            modality_length = WEIGHT_LENGTH; 
        } else  {
            std::cerr << "error in specify sensing modality" << std::endl; 
        }

        boost::numeric::ublas::matrix<float> obs_mat(modality_length, modality_length); 

        for (int i=0; i<modality_length; i++) {
            for (int j=0; j<modality_length; j++) {
                std::string tmp; 
                infile >> tmp; 
                obs_mat(i,j) = boost::lexical_cast<float>(tmp); 
            }
        }

        float quality; 
        evaluateMatrixQuality(obs_mat, quality); 

        std::cout << "\taction_name: " << action_name << "\t quality: " << quality << std::endl; 
        std::cout << std::endl; 

        if (quality_of_action.find(action_name) == quality_of_action.end()) {
            quality_of_action[action_name] = quality; 
            property_effective_[action_name] = property_name; 
        } 
        else if (quality < quality_of_action[action_name]) {
            quality_of_action[action_name] = quality; 
            property_effective_[action_name] = property_name; 
        }
        infile.close(); 
    }
     
    for (std::map<std::string, std::string>::iterator it = property_effective_.begin(); 
            it != property_effective_.end(); it++) {
        std::cout << "\taction: " << it->first << ", property: " << it->second << std::endl;  
    }
}

std::string FeatureSelector::getEffectiveProperty(std::string action_name) {
    if (property_effective_.find(action_name) == property_effective_.end())
        std::cout << "error in getting property of action: " << action_name << std::endl; 
    return property_effective_[action_name]; 
}
