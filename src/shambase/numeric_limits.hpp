// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file memory.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/aliases_float.hpp"
#include <limits>

namespace shambase{

    template<class T>
    struct primitive_type_info{
        static constexpr bool implemented = false;
    }

    namespace details {

        template<class T>
        struct numeric_limits{
            static constexpr bool implemented = false;
            
            inline constexpr static T max(){return {};}
            inline constexpr static T min(){return {};}
            inline constexpr static T epsilon(){return {};}
            inline constexpr static T infty(){return {};}
            inline constexpr static T infty_neg(){return {};}
            inline constexpr static T zero(){return {};}
        };

        template<>
        struct numeric_limits<f64>{
            using T = f64;
            static constexpr bool implemented = true;
            
            inline constexpr static T max(){return std::numeric_limits<T>::max();}
            inline constexpr static T min(){return {};}
            inline constexpr static T epsilon(){return {};}
            inline constexpr static T infty(){return {};}
            inline constexpr static T infty_neg(){return {};}
            inline constexpr static T zero(){return {};}
        };


    }

    template<class T>
    static constexpr T get_max(){
        static_assert(details::numeric_limits<T>::implemented, "this type was not implemented in shambase::details::numeric_limits" );
        return details::numeric_limits<T>::max();
    }

}