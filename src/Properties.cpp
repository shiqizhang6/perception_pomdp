
#include "Properties.h"

std::ostream& operator<<(std::ostream& stream, const Color& color) {
    if (color == RED) {
        stream << "Red"; 
    } else if (color == GREEN) {
        stream << "Green";
    } else if (color == BLUE) {
        stream << "Blue"; 
    } else {
        stream << "Unknown"; 
    }
    return stream; 
}

std::ostream& operator<<(std::ostream& stream, const Content& content) {
    if (content == CONTENT0) {
        stream << "Content0"; 
    } else if (content == CONTENT1) {
        stream << "Content1";
    } else if (content == CONTENT2) {
        stream << "Content2";
    } else if (content == CONTENT3) {
        stream << "Content3";
    } else {
        stream << "Unknown"; 
    }
    return stream; 
}

std::ostream& operator<<(std::ostream& stream, const Weight& weight) {
    if (weight == HEAVY) {
        stream << "Heavy";
    } else if (weight == MEDIUM) {
        stream << "Medium";
    } else if (wegiht == LIGHT) {
        stream << "Light";
    } else {
        stream << "Unknown";
    }
    return stream; 
}


