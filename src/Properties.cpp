
#include "Properties.h"

std::ostream& operator<<(std::ostream& stream, const Color& color) {
    if (color == RED) {
        stream << "red"; 
    } else if (color == GREEN) {
        stream << "green";
    } else if (color == BLUE) {
        stream << "blue"; 
    } else {
        stream << "unknown"; 
    }
    return stream; 
}

std::ostream& operator<<(std::ostream& stream, const Content& content) {
    if (content == CONTENT0) {
        stream << "content0"; 
    } else if (content == CONTENT1) {
        stream << "content1";
    } else if (content == CONTENT2) {
        stream << "content2";
    } else if (content == CONTENT3) {
        stream << "content3";
    } else {
        stream << "unknown"; 
    }
    return stream; 
}

std::ostream& operator<<(std::ostream& stream, const Weight& weight) {
    if (weight == HEAVY) {
        stream << "heavy";
    } else if (weight == MEDIUM) {
        stream << "medium";
    } else if (wegiht == LIGHT) {
        stream << "light";
    } else {
        stream << "unknown";
    }
    return stream; 
}


