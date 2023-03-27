// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"

namespace shambase {

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // product_accumulate
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr T product_accumulate(T v) noexcept {
        return v;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() * v.y();
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() * v.y() * v.z();
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() * v.y() * v.z() * v.w();
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.s0() * v.s1() * v.s2() * v.s3() * v.s4() * v.s5() * v.s6() * v.s7();
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.s0() * v.s1() * v.s2() * v.s3() * v.s4() * v.s5() * v.s6() * v.s7() * v.s8() *
               v.s9() * v.sA() * v.sB() * v.sC() * v.sD() * v.sE() * v.sF();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // all_component_are_negative
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool all_component_are_negative(T a) {
        return a < 0;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0) && (v.z() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0) && (v.z() < 0) && (v.w() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.s0() < 0) && (v.s1() < 0) && (v.s2() < 0) && (v.s3() < 0) && (v.s4() < 0) &&
               (v.s5() < 0) && (v.s6() < 0) && (v.s7() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.s0() < 0) && (v.s1() < 0) && (v.s2() < 0) && (v.s3() < 0) && (v.s4() < 0) &&
               (v.s5() < 0) && (v.s6() < 0) && (v.s7() < 0) && (v.s8() < 0) && (v.s9() < 0) &&
               (v.sA() < 0) && (v.sB() < 0) && (v.sC() < 0) && (v.sD() < 0) && (v.sE() < 0) &&
               (v.sF() < 0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_a_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_a_zero(T a) {
        return a == 0;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0) || (v.z() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0) || (v.z() == 0) || (v.w() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.s0() == 0) || (v.s1() == 0) || (v.s2() == 0) || (v.s3() == 0) || (v.s4() == 0) ||
               (v.s5() == 0) || (v.s6() == 0) || (v.s7() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.s0() == 0) || (v.s1() == 0) || (v.s2() == 0) || (v.s3() == 0) || (v.s4() == 0) ||
               (v.s5() == 0) || (v.s6() == 0) || (v.s7() == 0) || (v.s8() == 0) || (v.s9() == 0) ||
               (v.sA() == 0) || (v.sB() == 0) || (v.sC() == 0) || (v.sD() == 0) || (v.sE() == 0) ||
               (v.sF() == 0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_only_one_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_only_one_zero(T a) {
        return a == 0;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) != (v.y() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)} + int{(v.w() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} +
                        int{(v.s3() == 0)} + int{(v.s4() == 0)} + int{(v.s5() == 0)} +
                        int{(v.s6() == 0)} + int{(v.s7() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} +
                        int{(v.s3() == 0)} + int{(v.s4() == 0)} + int{(v.s5() == 0)} +
                        int{(v.s6() == 0)} + int{(v.s7() == 0)} + int{(v.s8() == 0)} +
                        int{(v.s9() == 0)} + int{(v.sA() == 0)} + int{(v.sB() == 0)} +
                        int{(v.sC() == 0)} + int{(v.sD() == 0)} + int{(v.sE() == 0)} +
                        int{(v.sF() == 0)};
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_at_most_one_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_at_most_one_zero(T a) {
        return true;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)} + int{(v.w() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} +
                        int{(v.s3() == 0)} + int{(v.s4() == 0)} + int{(v.s5() == 0)} +
                        int{(v.s6() == 0)} + int{(v.s7() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} +
                        int{(v.s3() == 0)} + int{(v.s4() == 0)} + int{(v.s5() == 0)} +
                        int{(v.s6() == 0)} + int{(v.s7() == 0)} + int{(v.s8() == 0)} +
                        int{(v.s9() == 0)} + int{(v.sA() == 0)} + int{(v.sB() == 0)} +
                        int{(v.sC() == 0)} + int{(v.sD() == 0)} + int{(v.sE() == 0)} +
                        int{(v.sF() == 0)};
    }

} // namespace shambase