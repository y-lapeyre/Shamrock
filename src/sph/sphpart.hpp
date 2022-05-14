#pragma once

#include "aliases.hpp"
#include "hipSYCL/sycl/libkernel/builtins.hpp"

inline constexpr f64 hfact = 1.2;

template<class flt>
inline flt rho_h(flt m, flt h){
    constexpr flt hfac = hfact;
    return m*(hfac/h)*(hfac/h)*(hfac/h);
}

template<class flt>
inline flt h_rho(flt m, flt rho){
    constexpr flt hfac = hfact;
    return hfac/sycl::pow(rho/m, 1/3);
}