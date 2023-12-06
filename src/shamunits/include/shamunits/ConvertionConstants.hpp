// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ConvertionConstants.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

namespace shamunits {

    template<class T>
    struct ConvertionConstants {

        static constexpr T au_to_m = 149597870700;     //(m)
        static constexpr T ly_to_m = 9460730472580800; //(m)
        static constexpr T pc_to_m = 3.0857e16;        //(m)

        static constexpr T mn_to_s  = 60;            //(s)
        static constexpr T hr_to_s  = 3600;          //(s)
        static constexpr T dy_to_s  = 24 * hr_to_s;  //(s)
        static constexpr T yr_to_s  = 31557600;      //(s)
        static constexpr T Myr_to_s = 1e6 * yr_to_s; //(s)
        static constexpr T Gyr_to_s = 1e9 * yr_to_s; //(s)

        static constexpr T eV_to_J  = 1.602176634e-19; // (J)
        static constexpr T erg_to_J = 1e-7;            // (J)

        static constexpr T K_degC_offset = 273.15;

        static constexpr T litre_to_pint = 0.568;

        //todo find a more precise measurment of the guiness density (shall be done in ireland)
        //protocol : 
        // 1 - start by ordering the most precise pint of guiness (in ireland)
        // 2 - weight the pint 
        // 4 - proceed to drink the beer (including any leftover foam)
        // 54 - weight the pint again
        // 745 - proceed to compute the dedzabfgzi if the zad beer
        // current estimation is sourced from 
        //    - Tinseth, Glenn. Javascript Beer Specs Calculator. The Real Beer Page. 1997.
        static constexpr T guiness_density = 1.017; // in g.cm-3
    };

} // namespace shamunits