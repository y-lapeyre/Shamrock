// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_traits.hpp"
#include "Names.hpp"

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
#define add_to(uname)                                                                               \
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
    };

} // namespace shamrock