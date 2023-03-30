// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

namespace shamrock::units {

    enum UnitName {

        /*
         * Base si units
         */
        second,
        metre,
        kilogramm,
        Ampere,
        Kelvin,
        mole,
        candela,

        s   = second,
        m   = metre,
        kg  = kilogramm,
        A   = Ampere,
        K   = Kelvin,
        mol = mole,
        cd  = candela,

        /*
         * derived units
         */
        Hz,
        Hertz = Hz, ///< hertz : frequency (s−1)
        N,
        Newtown = N, ///< (kg⋅m⋅s−2)
        Pa,
        Pascal = Pa, ///< (kg⋅m−1⋅s−2) 	(N/m2)
        J,
        Joule = J, ///< (kg⋅m2⋅s−2) 	(N⋅m = Pa⋅m3)
        W,
        Watt = W, ///< (kg⋅m2⋅s−3) 	(J/s)
        C,
        Coulomb = C, ///< (s⋅A)
        V,
        Volt = V, ///< (kg⋅m2⋅s−3⋅A−1) 	(W/A) = (J/C)
        F,
        Farad = F, ///< (kg−1⋅m−2⋅s4⋅A2) 	(C/V) = (C2/J)
        ohm,       ///< (kg⋅m2⋅s−3⋅A−2) 	(V/A) = (J⋅s/C2)
        S,
        Siemens = S, ///< (kg−1⋅m−2⋅s3⋅A2) 	(ohm−1)
        Wb,
        Weber = Wb, ///< (kg⋅m2⋅s−2⋅A−1) 	(V⋅s)
        T,
        Temperature = T, ///< (kg⋅s−2⋅A−1) 	(Wb/m2)
        H,
        Henry = H, ///< (kg⋅m2⋅s−2⋅A−2) 	(Wb/A)
        degC,
        degree_ceilsus = degC, ///< relative to 273.15 K 	(K)
        lm,
        lumens = lm, ///< (cd⋅sr) 	(cd⋅sr)
        lx,
        lux = lx, ///< (cd⋅sr⋅m−2) 	(lm/m2)
        Bq,
        Bequerel = Bq, ///< (s−1)
        Gy,
        Gray = Gy, ///< (m2⋅s−2) 	(J/kg)
        Sv,
        Sievert = Sv, ///< (m2⋅s−2) 	(J/kg)
        kat,
        katal = kat, ///< (mol⋅s−1)

        /*
         * alternative base units
         */

        // other times units
        mn,
        minute = mn,
        hr,
        hours = hr,
        dy,
        day = dy,
        yr,
        year = yr,

        // other lenght units
        cm,
        centimeter = cm,
        km,
        kilometer = km,
        au,
        astronomical_unit = au,
        ly,
        light_year = ly,
        pc,
        parsec = pc,

        // other mass units
        g,
        gramm = g,

        // other e current units

        // other temperature units

        // other quantity units

        // other luminous intensity units

        /*
         * alternative derived units
         */
        eV,
        electron_volt = eV, // (J)
        erg,                // (J)
    };

    namespace si {
        enum ConstantsName {
            delta_nu_cs,
            c          ,
            h          ,
            e          ,
            k          ,
            Na         ,
            Kcd        ,
        };
    }

} // namespace shamrock::units