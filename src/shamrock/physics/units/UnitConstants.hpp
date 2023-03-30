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

namespace shamrock::units {

    template<class T>
    class UnitConstants {

        static constexpr T pi = shambase::Constants<T>::pi;

        class Si {
            public:
            static constexpr T delta_nu_cs = 9192631770;      // (s-1)
            static constexpr T c           = 299792458;       // (m.s-1)
            static constexpr T h           = 6.62607015e-34;  // (J.s-1)
            static constexpr T e           = 1.602176634e-19; // (C)
            static constexpr T k           = 1.380649e-23;    // (J.K-1 )
            static constexpr T Na          = 6.02214076e23;   // (mol-1 )
            static constexpr T Kcd         = 683;             // (lm.W-1)
        };

        class Physics {
            public:
            static constexpr T G_si    = 6.6743015e-11;      // (N.m2.kg-2)
            static constexpr T hbar_si = 1.054571817e-34;    // (J.s-1)
            static constexpr T mu_0_si = 1.2566370621219e-6; //

            static constexpr T Z_0_si       = mu_0_si * Si::c;               //
            static constexpr T epsilon_0_si = 1 / (mu_0_si * Si::c * Si::c); //
            static constexpr T ke           = 1 / (4 * pi * epsilon_0_si);   //
        };

        class Convertion {
            public:
            static constexpr T au_m = 149597870700;     //(m)
            static constexpr T ly_m = 9460730472580800; //(m)
            static constexpr T pc_m = 3.0857e16;        //(m)

            static constexpr T mn_s = 60;        //(s)
            static constexpr T hr_s = 3600;      //(s)
            static constexpr T dy_s = 24 * hr_s; //(s)
            static constexpr T yr_s = 31557600;  //(s)

            static constexpr T earth_mass_kg   = 5.9722e24;  //(kg)
            static constexpr T jupiter_mass_kg = 1.898e27;   //(kg)
            static constexpr T sol_mass_kg     = 1.98847e30; //(kg)

            static constexpr T eV_J  = 1.602176634e-19; // (J)
            static constexpr T erg_J = 1e-7;            // (J)

            static constexpr T K_degC_offset = 273.15;
        };
    };

} // namespace shamrock::units