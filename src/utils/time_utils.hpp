#pragma once

#include "string_utils.hpp"
#include <chrono>
#include <string>
#include <vector>


inline std::string nanosec_to_time_str(double nanosec) {
    double sec_int = nanosec;

    std::string unit = "ns";

    if (sec_int > 2000) {
        unit = "us";
        sec_int /= 1000;
    }

    if (sec_int > 2000) {
        unit = "ms";
        sec_int /= 1000;
    }

    if (sec_int > 2000) {
        unit = "s";
        sec_int /= 1000;
    }

    if (sec_int > 2000) {
        unit = "ks";
        sec_int /= 1000;
    }

    if (sec_int > 2000) {
        unit = "Ms";
        sec_int /= 1000;
    }

    if (sec_int > 2000) {
        unit = "Gs";
        sec_int /= 1000;
    }

    return format("%4.2f", sec_int) + " " + unit;
} 


class Timer {
  public:
    std::chrono::steady_clock::time_point t_start, t_end;
    double sec;

    Timer(){};

    inline void start() { t_start = std::chrono::steady_clock::now(); }

    inline void end() {
        t_end = std::chrono::steady_clock::now();
        sec   = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
    }

    inline std::string get_time_str() {
        return nanosec_to_time_str(sec);
    }

    
};
