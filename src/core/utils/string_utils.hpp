// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file string_utils.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <fstream>
#include <string>
#include <cstdarg>

#include "aliases.hpp"
#include <cstdarg>



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

inline void write_string_to_file(std::string filename, std::string s) {
    std::ofstream myfile;
    myfile.open(filename);
    myfile << s;
    myfile.close();
}

//taken from https://en.cppreference.com/w/cpp/string/basic_string/replace
inline void replace_all(std::string& inout, std::string_view what, std::string_view with)
{
    for (std::string::size_type pos{};
         inout.npos != (pos = inout.find(what.data(), pos, what.length()));
         pos += with.length()) {
        inout.replace(pos, what.length(), with.data(), with.length());
    }
}

inline std::string increase_indent(std::string in){
    std::string out = in;
    replace_all(out, "\n", "\n    ");
    return "    " + out;
}

inline std::string trunc_str(std::string s , u32 max_len){

    if(max_len < 5) throw std::invalid_argument("max len should be above 4");

    if (s.size() > max_len) {
        return s.substr(0,max_len-5) + " ...";
    }else{
        return s;
    }

}

/**
 * @brief convert byte count to string
 * 
 * @param size 
 * @return std::string 
 */
inline std::string readable_sizeof(double size) {
    int i = 0;
    char buf[10];
    const char* units[] = {"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};

    if(size >= 0){
        while (size > 1024) {
            size /= 1024;
            i++;
        }
    }else{
        i = 9;
    }
    

    if(i > 8){
        sprintf(buf, "%s", "err val");
    }else{
        sprintf(buf, "%.2f %s", size, units[i]);
    }

    return std::string(buf);
}
