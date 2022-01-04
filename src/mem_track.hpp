#pragma once

#include <string>
#include <map>
#include <iostream>
#include <ostream>
#include <cstdio>
#include <sstream>

#include <cstring>


#define __FILENAME2__ std::string(strstr(__FILE__, "/src/") ? strstr(__FILE__, "/src/")+1  : __FILE__)
#define log_alloc_ln " ("+ __FILENAME2__ +":" + std::to_string(__LINE__) +")"


inline std::string ptr_to_str(void* ptr){
    std::stringstream strm;
    strm << ptr;
    return strm.str(); 
}

inline std::map<std::string,std::string> ptr_allocated;

inline void* log_new(void* ptr, std::string log){
    std::string ptr_loc = ptr_to_str(ptr);
    std::cout << "new : " << ptr_loc << " (" << log << ")\n";

    ptr_allocated[ptr_loc] = log;

    return ptr;
}

inline void log_delete(void* ptr, std::string log_){
    std::string ptr_loc = ptr_to_str(ptr);
    std::string log = ptr_allocated[ptr_loc];
    std::cout << "delete : " << ptr_loc << " (alloc : " << log << ") " << log_ << "\n";

    ptr_allocated.erase(ptr_loc);

}

inline void print_state_alloc(){
    std::cout << "---- allocated ----\n";
    for(auto obj: ptr_allocated) std::cout << "->" << obj.first << "("<< obj.second << ")\n";
    std::cout << "---- --------- ----\n";
}
