// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl.hpp"

namespace shambase {


    template<class T>
    class Constants{public:
        static constexpr T unity = 1;

        static constexpr T pi = 3.14159265358979323846;
        static constexpr T pi_square = 9.869604401089358;

        static constexpr T sqrt_2 = 1.4142135623730951;


        static constexpr T e = 2.718281828459045;
    };


}