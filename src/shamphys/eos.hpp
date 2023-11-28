// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sphkernels.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

namespace shamphys {

    template<class T>
    class EOS_Adiabatic{


        static constexpr T pressure(T gamma, T rho, T u){
            return (gamma - 1) * rho * u;
        }

        static constexpr T soundspeed(T gamma, T rho, T u) {
            return sycl::sqrt(gamma * eos_adiabatic(gamma, rho, u) / rho);
        }

        static constexpr T cs_from_p(T gamma, T rho, T P) {
            return sycl::sqrt(gamma * P / rho);
        }

    };


    template<class T>
    class EOS_Polytropic{


        static constexpr T pressure(T gamma, T K, T rho){
            return K * sycl::pow(rho, gamma);
        }

        static constexpr T soundspeed(T gamma, T K, T rho) {
            return sycl::sqrt(gamma * eos_adiabatic(gamma, K, rho) / rho);
        }

        static constexpr T polytropic_index(T n){
            return 1. + 1./n;
        }

    };


    template<class T>
    class EOS_LocallyIsothermal{

        static constexpr T soundspeed_sq(T cs0sq,T Rsq, T q){
            return cs0sq * sycl::pow(Rsq,q/2);
        }

        static constexpr T pressure(T cs0sq,T Rsq, T q, T rho){
            return soundspeed_sq(cs0sq, Rsq, q)*rho;
        }
    };



} // namespace shamphys