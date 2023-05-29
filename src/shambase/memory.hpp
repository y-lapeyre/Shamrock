// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "shambase/type_aliases.hpp"

namespace shambase {


    /**
     * @brief store a value of type T in a byte buffer
     * 
     * @tparam T 
     * @tparam AccU8 
     * @param acc 
     * @param ptr_write 
     * @param a 
     */
    template<class T,class AccU8>
    inline void store_u8(AccU8 & acc, u64 ptr_write, T a){
        constexpr u64 szT = sizeof(T);
        u8 *bytes = (u8 *)&a;
        #pragma unroll
        for(u64 i = 0; i < szT ; i++){
            acc[ptr_write+i] = bytes[i];
        }
    }

    /**
     * @brief load a value of type T from a byte buffer
     * 
     * @tparam T 
     * @tparam AccU8 
     * @param acc 
     * @param ptr_load 
     * @return T 
     */
    template<class T,class AccU8>
    inline T load_u8(AccU8 & acc, u64 ptr_load){
        constexpr u64 szT = sizeof(T);
        T ret;
        u8 *bytes = (u8 *)&ret;
        #pragma unroll
        for(u64 i = 0; i < szT ; i++){
            bytes[i] = acc[ptr_load + i];
        }
        return ret;
    }

    /**
     * @brief pointer cast store the value @param a in the pointer
     * 
     * @tparam T 
     * @tparam TAcc 
     * @param acc 
     * @param a 
     */
    template<class T, class TAcc>
    inline void store_conv(TAcc* acc, T a){
        T* ptr = (T*) acc;
        *ptr = a;
    }

    /**
     * @brief pointer cast load from a pointer
     * 
     * @tparam T 
     * @tparam TAcc 
     * @param acc 
     * @return T 
     */
    template<class T,class TAcc>
    inline T load_conv(TAcc* acc){
        T* ptr = (T*) acc;
        return *ptr;
    }

    /**
     * @brief Get reference to object held by the unique ptr, and throw if nothing is held
     * 
     * @tparam T 
     * @param ptr 
     * @return T& 
     */
    template<class T>
    inline T & get_check_ref(const std::unique_ptr<T> & ptr){
        if(! bool(ptr)){
            throw throw_with_loc<std::runtime_error>("the ptr does not hold anything");
        }
        return *ptr;
    }

}