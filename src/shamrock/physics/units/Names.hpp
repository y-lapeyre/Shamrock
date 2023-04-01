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
         * si derived units
         */

        mps, ///< meter per second (m.s-1)

        Hertz,       ///< hertz : frequency (s−1)
        Newtown,     ///< (kg⋅m⋅s−2)
        Pascal,      ///< (kg⋅m−1⋅s−2) 	(N/m2)
        Joule,       ///< (kg⋅m2⋅s−2) 	(N⋅m = Pa⋅m3)
        Watt,        ///< (kg⋅m2⋅s−3) 	(J/s)
        Coulomb,     ///< (s⋅A)
        Volt,        ///< (kg⋅m2⋅s−3⋅A−1) 	(W/A) = (J/C)
        Farad,       ///< (kg−1⋅m−2⋅s4⋅A2) 	(C/V) = (C2/J)
        Ohm,         ///< (kg⋅m2⋅s−3⋅A−2) 	(V/A) = (J⋅s/C2)
        Siemens,     ///< (kg−1⋅m−2⋅s3⋅A2) 	(ohm−1)
        Weber,       ///< (kg⋅m2⋅s−2⋅A−1) 	(V⋅s)
        Tesla, ///< (kg⋅s−2⋅A−1) 	(Wb/m2)
        Henry,       ///< (kg⋅m2⋅s−2⋅A−2) 	(Wb/A)
        lumens,      ///< (cd⋅sr) 	(cd⋅sr)
        lux,         ///< (cd⋅sr⋅m−2) 	(lm/m2)
        Bequerel,    ///< (s−1)
        Gray,        ///< (m2⋅s−2) 	(J/kg)
        Sievert,     ///< (m2⋅s−2) 	(J/kg)
        katal,       ///< (mol⋅s−1)

        Hz = Hertz,   
        N = Newtown,    
        Pa = Pascal,   
        J = Joule,   
        W = Watt,    
        C = Coulomb,    
        V = Volt,   
        F = Farad,    
        ohm = Ohm,  
        S = Siemens,    
        Wb = Weber,   
        T = Tesla,    
        H = Henry,  	
        lm = lumens,   
        lx = lux,   
        Bq = Bequerel,   
        Gy = Gray,   
        Sv = Sievert,   
        kat = katal,  

        /*
         * alternative base units
         */

        // other times units
        minute,
        hours,
        days,
        years,
        mega_years,
        giga_years,
        
        mn = minute,
        hr = hours,
        dy = days,
        yr = years,
        Myr = mega_years,
        Gyr = giga_years,

        // other lenght units
        nanometer,
        micrometer,
        millimeter,
        centimeter,
        kilometer,
        astronomical_unit,
        light_year,
        parsec,

        nm = nanometer,
        mu_m = micrometer,
        mm = millimeter,
        cm = centimeter,
        km = kilometer,
        au = astronomical_unit,
        ly = light_year,
        pc = parsec,

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
            c,
            h,
            e,
            k,
            Na,
            Kcd,
        };
    }

} // namespace shamrock::units