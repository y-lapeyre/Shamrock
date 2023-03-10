// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//
#pragma once
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



#include <fstream>
#include <string>
#include "aliases.hpp"
#include "shambase/stringUtils.hpp"
#include <cstdarg>

[[deprecated]]
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
 [[deprecated]]
inline std::string readable_sizeof(double size) {
    return shambase::readable_sizeof(size);
}
