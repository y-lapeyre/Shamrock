

/**
* \file logger.hpp
* blabla
*/


#pragma once 

#include <cstdarg>
#include <stdio.h>
#include <stdlib.h>
#include <string>

/**
 * @brief Logger class to pipe printf like outputs to files
 * 
 */
class Logger{
    public:

    /**
     * @brief log file name
     */
    std::string log_file_name;

    /**
     * @brief internal pointer to FILE object
     */
    FILE * log_file;

    /**
     * @brief Construct a new Logger object
     * 
     * @param filename filename which output would be piped to using log()
     */
    inline Logger(std::string filename){
        log_file_name = filename;
        log_file = fopen (log_file_name.c_str(), "w+");
    }

    /**
     * @brief Destroy the Logger object
     */
    inline ~Logger(){
        fclose(log_file);
    }

    /**
     * @brief printf like syntax to log to file
     * 
     * @param aFormat 
     * @param ... 
     */
    inline void log(const char* aFormat, ...){
        va_list argptr;
        va_start(argptr, aFormat);
        
        
        vfprintf(log_file, aFormat, argptr) ;
        
        
        va_end(argptr);
    
    }
    
};


