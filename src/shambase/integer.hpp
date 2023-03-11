// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"
#include <climits>
#include "shambase/type_traits.hpp"

namespace shambase {

    /**
     * @brief determine if v is a power of two
     * Warning : this function return true if v == 0
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     *
     * @tparam T the type
     * @param v
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline bool is_pow_of_two_fast(T v) {
        return (v & (v - 1)) == 0;
    }

    /**
     * @brief determine if v is a power of two and check if v==0
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     * @tparam T the type
     * @param v
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline bool is_pow_of_two(T v) {
        return v && !(v & (v - 1));
    }

    /**
     * @brief check if the sign of the two integers differs
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     *
     * @tparam T
     * @param a
     * @param b
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || std::is_signed_v<T>, int> = 0>
    inline bool sign_differ(T a, T b) {
        return ((a ^ b) < 0);
    }

    /**
     * @brief swap two values using xor
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     * 
     * @tparam T 
     * @param a 
     * @param b 
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline void xor_swap(T &a, T &b) {
        a ^= b;
        b ^= a;
        a ^= b;
    };

    /**
     * @brief round up to the next power of two
     * 
     * @tparam T 
     * @param v 
     * @return constexpr T 
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || has_bitlen_v<T, 32> || (!std::is_signed_v<T>), int> = 0>
    inline constexpr T roundup_pow2 (T v) noexcept {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
    };

    /**
     * @brief round up to the next power of two
     * 
     * @tparam T 
     * @param v 
     * @return constexpr T 
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || has_bitlen_v<T, 64> || (!std::is_signed_v<T>), int> = 0>
    inline constexpr T roundup_pow2 (T v) noexcept {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v++;
    };

} // namespace shambase