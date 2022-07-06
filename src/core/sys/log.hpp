#pragma once 

#include <string>
#include <iostream>


namespace logger {

    void print(){}

    template <typename T, typename... Types>
    void print(T var1, Types... var2)
    {
        std::cout << var1 << " ";
    
        print(var2...);
    }
 

    

    template <typename... Types>
    void normal(std::string module_name, Types... var2){
        std::cout << "["+module_name+"]";
        print(var2...);
    }

    template <typename... Types>
    void normal_ln(std::string module_name, Types... var2){
        std::cout << module_name;
        print(var2...);
        std::cout << std::endl;
    }
}