// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "Names.hpp"
#include "shambase/floats.hpp"
#include "shambase/type_traits.hpp"
#include "shamrock/physics/units/ConvertionConstants.hpp"

#define addget(uname)                                                                              \
    template<                                                                                      \
        UnitPrefix pref = None,                                                                    \
        units::UnitName u,                                                                         \
        i32 power                                = 1,                                              \
        std::enable_if_t<u == units::uname, int> = 0>                                              \
    inline constexpr T get() noexcept

#define PREF get_prefix_val<pref>()
#define Uget(unitname, mult_pow) get<units::unitname, (mult_pow)*power>()
#define Cget(constant_name, mult_pow) shambase::pow_constexpr_fast_inv<(mult_pow)*power>(constant_name,T(1) / constant_name)

namespace shamrock {


    template<class T>
    class UnitSystem {

        template<i32 power>
        inline static constexpr T pow(T a, T a_inv) noexcept {
            return shambase::pow_constexpr_fast_inv<power>(a, a_inv);
        }

        using Uconvert = ConvertionConstants<T>;

        public:
        const T s, m, kg, A, K, mol, cd;

        const T s_inv, m_inv, kg_inv, A_inv, K_inv, mol_inv, cd_inv;

        UnitSystem(
            T unit_time,
            T unit_lenght,
            T unit_mass,
            T unit_current,
            T unit_temperature,
            T unit_qte,
            T unit_lumint
        )
            : s(1 / unit_time), m(1 / unit_lenght), kg(1 / unit_mass), A(1 / unit_current),
              K(1 / unit_temperature), mol(1 / unit_qte), cd(1 / unit_lumint), s_inv(unit_time),
              m_inv(unit_lenght), kg_inv(unit_mass), A_inv(unit_current), K_inv(unit_temperature),
              mol_inv(unit_qte), cd_inv(unit_lumint) {}

        // clang-format off
        addget(second)    { return PREF* pow<power>(s  , s_inv);   }
        addget(metre)     { return PREF* pow<power>(m  , m_inv);   }
        addget(kilogramm) { return PREF* pow<power>(kg , kg_inv);  }
        addget(Ampere)    { return PREF* pow<power>(A  , A_inv);   }
        addget(Kelvin)    { return PREF* pow<power>(K  , K_inv);   }
        addget(mole)      { return PREF* pow<power>(mol, mol_inv); }
        addget(candela)   { return PREF* pow<power>(cd , cd_inv);  }
        
        addget(Hertz)   { return PREF* Uget(s, -1); }
        addget(mps)     { return PREF* Uget(m, 1)       * Uget(s, -1); }
        addget(Newtown) { return PREF* Uget(kg, 1)      * Uget(m, 1)  * Uget(s, -2); }
        addget(Pascal)  { return PREF* Uget(kg, 1)      * Uget(m, -1) * Uget(s, -2); }
        addget(Joule)   { return PREF* Uget(Newtown, 1) * Uget(m, 1); }
        addget(Watt)    { return PREF* Uget(Joule, 1)   * Uget(s, -1); }
        addget(Coulomb) { return PREF* Uget(s, 1)       * Uget(A, 1); }
        addget(Volt)    { return PREF* Uget(Watt, 1)    * Uget(A, -1); }
        addget(Farad)   { return PREF* Uget(Coulomb, 1) * Uget(Volt, -1); }
        addget(Ohm)     { return PREF* Uget(Volt, 1)    * Uget(Ampere, -1); }
        addget(Siemens) { return PREF* Uget(Ohm, -1); }
        addget(Weber)   { return PREF* Uget(Volt, 1)    * Uget(second, 1); }
        addget(Tesla)   { return PREF* Uget(Weber, 1)   * Uget(m, -2); }
        addget(Henry)   { return PREF* Uget(Weber, 1)   * Uget(A, -1); }
        addget(lumens)  { return PREF* Uget(candela, 1); }
        addget(lux)     { return PREF* Uget(lumens, 1)  * Uget(m, -2); }
        addget(Bequerel){ return PREF* Uget(s, -1); }
        addget(Gray)    { return PREF* Uget(m, 2)       *Uget(s, -2)  ; }
        addget(Sievert) { return PREF* Uget(m, 2)       *Uget(s, -2)  ; }
        addget(katal)   { return PREF* Uget(mol, 1)     *Uget(s, -1)  ; }
        

        // alternative base units
        addget(minute) { return PREF* Uget(s, 1) * Cget(Uconvert::mn_to_s, 1); }
        addget(hours)  { return PREF* Uget(s, 1) * Cget(Uconvert::hr_to_s, 1); }
        addget(days)   { return PREF* Uget(s, 1) * Cget(Uconvert::dy_to_s, 1); }
        addget(years)  { return PREF* Uget(s, 1) * Cget(Uconvert::yr_to_s, 1); }

        addget(astronomical_unit) { return PREF* Uget(m, 1) * Cget(Uconvert::au_to_m, 1); }
        addget(light_year)        { return PREF* Uget(m, 1) * Cget(Uconvert::ly_to_m, 1); }
        addget(parsec)            { return PREF* Uget(m, 1) * Cget(Uconvert::pc_to_m, 1); }

        addget(eV) {return PREF* Uget(Joule, 1) * Cget(Uconvert::eV_to_J,1);}
        addget(erg) {return PREF* Uget(Joule, 1) * Cget(Uconvert::erg_to_J,1);}

        // clang-format on


        template<units::UnitName u, i32 power>
        inline constexpr T to() {
            return get<u, -power>();
        }
    };

} // namespace shamrock

#undef addget
#undef PREF
#undef Uget
#undef Cget