
#ifndef UTILITIES_H
#define UTILITIES_H

#include "Simulator.h"


bool sameProperties(StateNonTerminal *s1, StateNonTerminal *s2) {
    return s1->color_ == s2->color_ and s1->content_ == s2->content_ and s1->weight_ == s2->weight_; 
}


#endif 
