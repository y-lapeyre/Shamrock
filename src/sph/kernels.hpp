#pragma once 

#include "aliases.hpp"

inline constexpr f64 PI = 3.141592653589793238463;

namespace sph {
namespace kernels {

// 3d kernels only

template <class flt_type> class M4 {public:
    inline static constexpr flt_type Rkern = 2;

    inline static constexpr flt_type norm = 1 / PI;

    inline static flt_type f(flt_type q) {

        constexpr flt_type div3_4 = (3. / 4.);
        constexpr flt_type div3_2 = (3. / 2.);
        constexpr flt_type div1_4 = (1. / 4.);

        if (q < 1) {
            return 1 + q * q * (div3_4 * q - div3_2);
        } else if (q < 2) {
            return div1_4 * (2 - q) * (2 - q) * (2 - q);
        } else
            return 0;
    }

    inline static flt_type df(flt_type q) {

        constexpr flt_type div9_4 = (9. / 4.);
        constexpr flt_type div3_4 = (3. / 4.);

        if (q < 1) {
            return -3 * q + div9_4 * q * q;
        } else if (q < 2) {
            return -3 + 3 * q - div3_4 * q * q;
        } else
            return 0;
    }

    inline static flt_type W(flt_type r, flt_type h) { return norm * f(r / h) / (h * h * h); }

    inline static flt_type dW(flt_type r, flt_type h) { return norm * df(r / h) / (h * h * h * h); }

    inline static flt_type dhW(flt_type r, flt_type h) {
        return -(norm) * (3 * f(r / h) + (r / h) * df(r / h)) / (h * h * h * h);
    }
};

} // namespace kernels
} // namespace sph