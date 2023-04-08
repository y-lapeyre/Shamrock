// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "shambase/floats.hpp"
#include <stdexcept>
#include <unordered_map>

#define XMAC_UNITS                                                                                 \
    /*base units*/                                                                                 \
    X1(second, s)                                                                                  \
    X1(metre, m)                                                                                   \
    X1(kilogramm, kg)                                                                              \
    X1(Ampere, A)                                                                                  \
    X1(Kelvin, K)                                                                                  \
    X1(mole, mol)                                                                                  \
    X1(candela, cd)                                                                                \
    /*derived units*/                                                                              \
    X1(Hertz, Hz)    /* hertz : frequency (s−1) */                                               \
    X1(Newtown, N)   /* (kg⋅m⋅s−2)*/                                                         \
    X1(Pascal, Pa)   /* (kg⋅m−1⋅s−2) 	(N/m2)*/                                             \
    X1(Joule, J)     /* (kg⋅m2⋅s−2) 	(N⋅m = Pa⋅m3)*/                                     \
    X1(Watt, W)      /* (kg⋅m2⋅s−3) 	(J/s)*/                                                 \
    X1(Coulomb, C)   /* (s⋅A)*/                                                                  \
    X1(Volt, V)      /* (kg⋅m2⋅s−3⋅A−1) 	(W/A) = (J/C)*/                                 \
    X1(Farad, F)     /* (kg−1⋅m−2⋅s4⋅A2) 	(C/V) = (C2/J)*/                               \
    X1(Ohm, ohm)     /* (kg⋅m2⋅s−3⋅A−2) 	(V/A) = (J⋅s/C2)*/                            \
    X1(Siemens, S)   /* (kg−1⋅m−2⋅s3⋅A2) 	(ohm−1)*/                                    \
    X1(Weber, Wb)    /* (kg⋅m2⋅s−2⋅A−1) 	(V⋅s)*/                                       \
    X1(Tesla, T)     /* (kg⋅s−2⋅A−1) 	(Wb/m2)*/                                            \
    X1(Henry, H)     /* (kg⋅m2⋅s−2⋅A−2) 	(Wb/A)*/                                        \
    X1(lumens, lm)   /* (cd⋅sr) 	(cd⋅sr)*/                                                     \
    X1(lux, lx)      /* (cd⋅sr⋅m−2) 	(lm/m2)*/                                               \
    X1(Bequerel, Bq) /* (s−1)*/                                                                  \
    X1(Gray, Gy)     /* (m2⋅s−2) 	(J/kg)*/                                                     \
    X1(Sievert, Sv)  /* (m2⋅s−2) 	(J/kg)*/                                                     \
    X1(katal, kat)   /* (mol⋅s−1) */                                                           \
    /*relative units*/                                                                             \
    X1(minutes, mn)                                                                                \
    X1(hours, hr)                                                                                  \
    X1(days, dy)                                                                                   \
    X1(years, yr)                                                                                  \
    X1(astronomical_unit, au)                                                                      \
    X1(light_year, ly)                                                                             \
    X1(parsec, pc)                                                                                 \
    X1(electron_volt, eV)                                                                          \
    X1(ergs, erg)






namespace shamrock {

    enum UnitPrefix {
        tera  = 12,  // e12
        giga  = 9,   // e9
        mega  = 6,   // e6
        kilo  = 3,   // e3
        hecto = 2,   // e2
        deca  = 1,   // e1
        None  = 0,   // 1
        deci  = -1,  // e-1
        centi = -2,  // e-2
        milli = -3,  // e-3
        micro = -6,  // e-6
        nano  = -9,  // e-9
        pico  = -12, // e-12
        femto = -15, // e-15
    };

    template<class T,UnitPrefix p>
    inline constexpr T get_prefix_val() {
        return shambase::pow_constexpr_fast_inv<p, T>(10, 1e-1);
    }

    inline const std::string get_prefix_str(UnitPrefix p) {
        switch (p) {
        case tera: return "T"; break;
        case giga: return "G"; break;
        case mega: return "M"; break;
        case kilo: return "k"; break;
        case hecto: return "x100"; break;
        case deca: return "x10"; break;
        case None: return ""; break;
        case deci: return "/10"; break;
        case centi: return "c"; break;
        case milli: return "m"; break;
        case micro: return "mu"; break;
        case nano: return "n"; break;
        case pico: return "p"; break;
        case femto: return "f"; break;
        }
        return "";
    }

    namespace units {

        // clang-format off
        enum UnitName {
            #define X1(longname, shortname) longname, shortname = longname,
            XMAC_UNITS
            #undef X1
        };

        static const std::unordered_map<std::string, UnitName> map_name_to_unit{
            #define X1(longname, shortname) {#longname, longname}, {#shortname, shortname},
            XMAC_UNITS
            #undef X1
        };

        static const std::unordered_map<UnitName, std::string> map_u_to_name = {
            #define X1(longname, shortname) {shortname, #shortname},
            XMAC_UNITS
            #undef X1
        };
        // clang-format on

        inline const std::string get_unit_name(UnitName p) {

            map_u_to_name.find(p);

            if (auto search = map_u_to_name.find(p); search != map_u_to_name.end()) {
                return search->second;
            }

            return "[Unknown Unit name]";
        }

        inline const UnitName unit_from_name(std::string p) {

            map_name_to_unit.find(p);

            if (auto search = map_name_to_unit.find(p); search != map_name_to_unit.end()) {
                return search->second;
            }

            shambase::throw_with_loc<std::invalid_argument>("this unit name is unknown");
            return s; // to silence a warning
        }

    } // namespace units

} // namespace shamrock