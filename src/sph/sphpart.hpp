#pragma once

#include "aliases.hpp"

inline constexpr f64 hfact = 1.2;

template<class flt>
inline flt rho_h(flt m, flt h){
    constexpr flt hfac = hfact;
    return m*(hfac/h)*(hfac/h)*(hfac/h);
}