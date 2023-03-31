// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/Constants.hpp"
#include "shambase/type_aliases.hpp"

namespace shamrock {

    template<class T>
    struct Constants {

        static constexpr T pi = shambase::Constants<T>::pi;        
        static constexpr T pi_square = pi*pi;

        struct SiBase {
            static constexpr T delta_nu_cs = 9192631770;      // (s-1)
            static constexpr T c           = 299792458;       // (m.s-1)
            static constexpr T h           = 6.62607015e-34;  // (J.s-1)
            static constexpr T e           = 1.602176634e-19; // (C)
            static constexpr T k           = 1.380649e-23;    // (J.K-1 )
            static constexpr T Na          = 6.02214076e23;   // (mol-1 )
            static constexpr T Kcd         = 683;             // (lm.W-1)
        };

        static constexpr T fine_structure        = 0.0072973525693;
        static constexpr T proton_electron_ratio = 1836.1526734311;
        static constexpr T electron_proton_ratio = 1 / proton_electron_ratio;

        struct Si {


            static constexpr T c_sq           = SiBase::c * SiBase::c;       // 

            static constexpr T G         = 6.6743015e-11;                      // (N.m2.kg-2)
            static constexpr T hbar      = 1.054571817e-34;                    // (J.s-1)
            static constexpr T mu_0      = 1.2566370621219e-6;                 //
            static constexpr T Z_0       = mu_0 * SiBase::c;                   //
            static constexpr T epsilon_0 = 1 / (Z_0 * SiBase::c); //
            static constexpr T ke        = 1 / (4 * pi * epsilon_0);           //

            static constexpr T proton_mass   = 1.67262192e-27;                      //(kg)
            static constexpr T electron_mass = proton_mass * electron_proton_ratio; //(kg)
            static constexpr T earth_mass     = 5.9722e24;                           //(kg)
            static constexpr T jupiter_mass   = 1.898e27;                            //(kg)
            static constexpr T sol_mass       = 1.98847e30;                          //(kg)
            static constexpr T planck_mass   = 2.17643424e-8; //(kg)

            static constexpr T planck_lenght = 1.61625518e-35; //(kg)
        };

        static constexpr T au_to_m = 149597870700;     //(m)
        static constexpr T ly_to_m = 9460730472580800; //(m)
        static constexpr T pc_to_m = 3.0857e16;        //(m)

        static constexpr T mn_to_s = 60;           //(s)
        static constexpr T hr_to_s = 3600;         //(s)
        static constexpr T dy_to_s = 24 * hr_to_s; //(s)
        static constexpr T yr_to_s = 31557600;     //(s)

        static constexpr T eV_to_J  = 1.602176634e-19; // (J)
        static constexpr T erg_to_J = 1e-7;            // (J)

        static constexpr T K_degC_offset = 273.15;
    };

} // namespace shamrock