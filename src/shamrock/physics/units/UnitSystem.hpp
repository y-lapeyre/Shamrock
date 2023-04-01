// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "Names.hpp"
#include "shambase/type_traits.hpp"
#include "shamrock/physics/Constants.hpp"

namespace shamrock {

    template<class T>
    struct UnitSystem {

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
              K(1 / unit_temperature), mol(1 / unit_qte), cd(1 / unit_lumint),

              s_inv(unit_time), m_inv(unit_lenght), kg_inv(unit_mass), A_inv(unit_current),
              K_inv(unit_temperature), mol_inv(unit_qte), cd_inv(unit_lumint) {}

#define addget(uname)                                                                              \
    template<units::UnitName u, std::enable_if_t<u == units::uname, int> = 0>                      \
    inline constexpr T get() noexcept
#define add_to(uname)                                                                              \
    template<units::UnitName u, std::enable_if_t<u == units::uname, int> = 0>                      \
    inline constexpr T to() noexcept

        addget(second) { return s; }
        add_to(second) { return s_inv; }

        addget(metre) { return m; }
        add_to(metre) { return m_inv; }

        addget(kilogramm) { return kg; }
        add_to(kilogramm) { return kg_inv; }

        addget(Ampere) { return A; }
        add_to(Ampere) { return A_inv; }

        addget(Kelvin) { return K; }
        add_to(Kelvin) { return K_inv; }

        addget(mole) { return mol; }
        add_to(mole) { return mol_inv; }

        addget(candela) { return cd; }
        add_to(candela) { return cd_inv; }

        addget(Hertz){return s_inv;}
        add_to(Hertz){return s;}

        addget(mps){return m*s_inv;}
        add_to(mps){return m_inv*s;}

        addget(Newtown){return kg*m*s_inv*s_inv;}
        add_to(Newtown){return kg_inv*m_inv*s*s;}

        addget(Pascal){return kg*m_inv*s_inv*s_inv;}
        add_to(Pascal){return kg_inv*m*s*s;}

        addget(Joule){return get<units::Newtown>()*m;}
        add_to(Joule){return to<units::Newtown>()*m_inv;}

        addget(Watt){return get<units::Joule>()*s_inv;}
        add_to(Watt){return to<units::Joule>()*s;}

        addget(Coulomb){return s*A;}
        add_to(Coulomb){return s_inv*A_inv;}

        addget(Volt){return get<units::Watt>()*A_inv;}
        add_to(Volt){return to<units::Watt>()*A;}

        addget(Farad){return get<units::Coulomb>()*to<units::Volt>();}
        add_to(Farad){return to<units::Coulomb>()*get<units::Volt>();}

        addget(Ohm){return get<units::Volt>()*to<units::Ampere>();}
        add_to(Ohm){return to<units::Volt>()*get<units::Ampere>();}

        addget(Siemens){return to<units::Ohm>();}
        add_to(Siemens){return get<units::Ohm>();}

        addget(Weber){return get<units::Volt>()*get<units::second>();}
        add_to(Weber){return to<units::Volt>()*to<units::second>();}

        addget(Tesla){return get<units::Weber>()*m_inv*m_inv;}
        add_to(Tesla){return to<units::Weber>()*m*m;}

        addget(Henry){return get<units::Weber>()*to<units::Ampere>();}
        add_to(Henry){return to<units::Weber>()*get<units::Ampere>();}

        addget(lumens){return get<units::candela>();}
        add_to(lumens){return to<units::candela>();}

        addget(lux){return get<units::lumens>()*m_inv*m_inv;}
        add_to(lux){return to<units::lumens>()*m*m;}

        addget(Bequerel){return to<units::second>();}
        add_to(Bequerel){return get<units::second>();}

        addget(Gray){return m*m*s_inv*s_inv;}
        add_to(Gray){return m_inv*m_inv*s*s;}

        addget(Sievert){return m*m*s_inv*s_inv;}
        add_to(Sievert){return m_inv*m_inv*s*s;}

        addget(katal){return mol*s_inv;}
        add_to(katal){return mol_inv*s;}


        // alternative base units

        addget(minute){return s*Constants<T>::mn_to_s;}
        add_to(minute){return s_inv/Constants<T>::mn_to_s;}

        addget(hours){return s*Constants<T>::hr_to_s;}
        add_to(hours){return s_inv/Constants<T>::hr_to_s;}

        addget(days){return s*Constants<T>::dy_to_s;}
        add_to(days){return s_inv/Constants<T>::dy_to_s;}

        addget(years){return s*Constants<T>::yr_to_s;}
        add_to(years){return s_inv/Constants<T>::yr_to_s;}

        addget(mega_years){return s*Constants<T>::Myr_to_s;}
        add_to(mega_years){return s_inv/Constants<T>::Myr_to_s;}

        addget(giga_years){return s*Constants<T>::Gyr_to_s;}
        add_to(giga_years){return s_inv/Constants<T>::Gyr_to_s;}





        addget(nanometer){return m*1e-9;}
        add_to(nanometer){return m_inv*1e9;}
        
        addget(micrometer){return m*1e-6;}
        add_to(micrometer){return m_inv*1e6;}
        
        addget(millimeter){return m*1e-3;}
        add_to(millimeter){return m_inv*1e3;}
        
        addget(centimeter){return m*1e-2;}
        add_to(centimeter){return m_inv*1e2;}
        
        addget(kilometer){return m*1e3;}
        add_to(kilometer){return m_inv*1e-3;}
        
        addget(astronomical_unit){return m*Constants<T>::au_to_m;}
        add_to(astronomical_unit){return m_inv/Constants<T>::au_to_m;}
        
        addget(light_year){return m*Constants<T>::ly_to_m;}
        add_to(light_year){return m_inv/Constants<T>::ly_to_m;}
        
        addget(parsec){return m*Constants<T>::pc_to_m;}
        add_to(parsec){return m_inv/Constants<T>::pc_to_m;}

    };

} // namespace shamrock