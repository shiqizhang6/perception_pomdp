
#include "Simulator.h"

Simulator::Simulator(int state_num, int action_num) {
    for (int i=0; i<state_num; i++) {
        State s = State(i); 
        states.push_back(s); 
    }
    for (int i=0; i<action_num; i++) {
        Action a = Action(i);
        actions.push_back(a);
    }

}
