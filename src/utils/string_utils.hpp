#pragma once

#include <string>
#include <fstream>

inline std::string format(const char *fmt...) {
    std::string s{};
    va_list args, args2;
    va_start(args, fmt);
    va_copy(args2, args);
 
    s.resize(vsnprintf(nullptr, 0, fmt, args2) + 1);
    va_end(args2);
    vsprintf(s.data(), fmt, args);
    va_end(args);
    s.pop_back();
    return s;
}

inline void write_string_to_file(std::string filename,std::string s){
    std::ofstream myfile;
    myfile.open (filename);
    myfile << s;
    myfile.close();
}