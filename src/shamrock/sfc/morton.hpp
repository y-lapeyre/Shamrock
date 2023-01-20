// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file morton.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Morton curve implementation
 * @version 0.1
 * @date 2022-03-03
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "aliases.hpp"
#include "bmi.hpp"
#include <type_traits>


namespace shamrock::sfc {

        
    template<class morton_repr, u32 dim>
    class MortonCodes{};



    template<> class MortonCodes<u32, 3>{public:

        using int_vec_repr_base                    = u16;
        using int_vec_repr                         = u16_3;
        static constexpr int_vec_repr_base max_val = 1024 - 1;

        static u64 icoord_to_morton(u64 x, u64 y, u64 z){
            u32 xx = bmi::expand_bits<u32, 2>((u32)x);
            u32 yy = bmi::expand_bits<u32, 2>((u32)y);
            u32 zz = bmi::expand_bits<u32, 2>((u32)z);
            return xx * 4 + yy * 2 + zz;
        }

        template<class flt>
        inline static u64 coord_to_morton(flt x, flt y, flt z){

            constexpr bool ok_type = std::is_same<flt,f32>::value || std::is_same<flt,f64>::value;
            static_assert(ok_type, "unknown input type");

            if constexpr (std::is_same<flt,f32>::value){
                
                x = sycl::fmin(sycl::fmax(x * 1024.F, 0.F), 1024.F - 1.F);
                y = sycl::fmin(sycl::fmax(y * 1024.F, 0.F), 1024.F - 1.F);
                z = sycl::fmin(sycl::fmax(z * 1024.F, 0.F), 1024.F - 1.F);

                return icoord_to_morton(x, y, z);

            }else if constexpr (std::is_same<flt,f64>::value){

                x = sycl::fmin(sycl::fmax(x * 1024., 0.), 1024. - 1.);
                y = sycl::fmin(sycl::fmax(y * 1024., 0.), 1024. - 1.);
                z = sycl::fmin(sycl::fmax(z * 1024., 0.), 1024. - 1.);

                return icoord_to_morton(x, y, z);

            }
        }

        inline static u16_3 morton_to_icoord(u32 morton) {

            u16_3 pos;
            pos.s0() = (u16)bmi::contract_bits<u32, 2>((morton & 0x24924924U) >> 2U);
            pos.s1() = (u16)bmi::contract_bits<u32, 2>((morton & 0x12492492U) >> 1U);
            pos.s2() = (u16)bmi::contract_bits<u32, 2>((morton & 0x09249249U) >> 0U);

            return pos;
        }

        inline static u16_3 get_offset(u32 clz_) {
            u16_3 mx;
            mx.s0() = 1024U >> ((clz_ - 0) / 3);
            mx.s1() = 1024U >> ((clz_ - 1) / 3);
            mx.s2() = 1024U >> ((clz_ - 2) / 3);
            return mx;
        }

    };


    template<> class MortonCodes<u64, 3>{public:

        using int_vec_repr_base                    = u32;
        using int_vec_repr                         = u32_3;
        static constexpr int_vec_repr_base max_val = 2097152 - 1;

        inline static u64 icoord_to_morton(u64 x, u64 y, u64 z){
            u64 xx = bmi::expand_bits<u64, 2>((u64)x);
            u64 yy = bmi::expand_bits<u64, 2>((u64)y);
            u64 zz = bmi::expand_bits<u64, 2>((u64)z);
            return xx * 4 + yy * 2 + zz;
        }

        template<class flt>
        inline static u64 coord_to_morton(flt x, flt y, flt z){

            constexpr bool ok_type = std::is_same<flt,f32>::value || std::is_same<flt,f64>::value;
            static_assert(ok_type, "unknown input type");

            if constexpr (std::is_same<flt,f32>::value){

                x = sycl::fmin(sycl::fmax(x * 2097152.F, 0.F), 2097152.F - 1.F);
                y = sycl::fmin(sycl::fmax(y * 2097152.F, 0.F), 2097152.F - 1.F);
                z = sycl::fmin(sycl::fmax(z * 2097152.F, 0.F), 2097152.F - 1.F);

                return icoord_to_morton(x, y, z);

            }else if constexpr (std::is_same<flt,f64>::value){

                x = sycl::fmin(sycl::fmax(x * 2097152., 0.), 2097152. - 1.);
                y = sycl::fmin(sycl::fmax(y * 2097152., 0.), 2097152. - 1.);
                z = sycl::fmin(sycl::fmax(z * 2097152., 0.), 2097152. - 1.);

                return icoord_to_morton(x, y, z);

            }
        }

        inline static int_vec_repr morton_to_icoord(u64 morton) {

            u32_3 pos;
            pos.x() = bmi::contract_bits<u64, 2>((morton & 0x4924924924924924U) >> 2U);
            pos.y() = bmi::contract_bits<u64, 2>((morton & 0x2492492492492492U) >> 1U);
            pos.z() = bmi::contract_bits<u64, 2>((morton & 0x1249249249249249U) >> 0U);

            return pos;
        }

        inline static int_vec_repr get_offset(u32 clz_) {
            u32_3 mx;
            mx.s0() = 2097152U >> ((clz_ + 1) / 3);
            mx.s1() = 2097152U >> ((clz_ - 0) / 3);
            mx.s2() = 2097152U >> ((clz_ - 1) / 3);
            return mx;
        }

        

    };




} // namespace shamrock::sfc

//%Impl status : Good

namespace morton_3d {

    /** 
     * @brief Helper struct to get types corresponding to a morton code representation
     * @tparam morton_repr u32 (32 bits) or u64 (64 bits) 
     */
    template <class morton_repr> struct [[deprecated]] morton_types {
        using int_vec_repr_base = std::void_t<>;
        using int_vec_repr      = std::void_t<>;
    };

    template <> struct [[deprecated]] morton_types<u32> {
        using int_vec_repr_base                    = u16;
        using int_vec_repr                         = u16_3;
        static constexpr int_vec_repr_base max_val = 1024 - 1;

        // not possible yet in hipsycl
        // static constexpr int_vec_repr max_vec = int_vec_repr{max_val};
    };

    template <> struct [[deprecated]] morton_types<u64> {
        using int_vec_repr_base                    = u32;
        using int_vec_repr                         = u32_3;
        static constexpr int_vec_repr_base max_val = 2097152 - 1;

        // not possible yet in hipsycl
        // static constexpr int_vec_repr max_vec = int_vec_repr{max_val};
    };

    template <class morton_prec, class fp_prec> [[deprecated]] morton_prec coord_to_morton(fp_prec x, fp_prec y, fp_prec z);

    template <class morton_prec> [[deprecated]] typename morton_types<morton_prec>::int_vec_repr morton_to_ipos(morton_prec morton);

    template <class morton_prec> [[deprecated]] typename morton_types<morton_prec>::int_vec_repr get_offset(u32 clz_);

    template <> inline u64 coord_to_morton<u64, f64>(f64 x, f64 y, f64 z) {
        x = sycl::fmin(sycl::fmax(x * 2097152., 0.), 2097152. - 1.);
        y = sycl::fmin(sycl::fmax(y * 2097152., 0.), 2097152. - 1.);
        z = sycl::fmin(sycl::fmax(z * 2097152., 0.), 2097152. - 1.);

        u64 xx = shamrock::sfc::bmi::expand_bits<u64, 2>((u64)x);
        u64 yy = shamrock::sfc::bmi::expand_bits<u64, 2>((u64)y);
        u64 zz = shamrock::sfc::bmi::expand_bits<u64, 2>((u64)z);
        return xx * 4 + yy * 2 + zz;
    }

    template <> inline u64 coord_to_morton<u64, f32>(f32 x, f32 y, f32 z) {
        x = sycl::fmin(sycl::fmax(x * 2097152.F, 0.F), 2097152.F - 1.F);
        y = sycl::fmin(sycl::fmax(y * 2097152.F, 0.F), 2097152.F - 1.F);
        z = sycl::fmin(sycl::fmax(z * 2097152.F, 0.F), 2097152.F - 1.F);

        u64 xx = shamrock::sfc::bmi::expand_bits<u64, 2>((u64)x);
        u64 yy = shamrock::sfc::bmi::expand_bits<u64, 2>((u64)y);
        u64 zz = shamrock::sfc::bmi::expand_bits<u64, 2>((u64)z);
        return xx * 4 + yy * 2 + zz;
    }

    template <> inline u32 coord_to_morton<u32, f64>(f64 x, f64 y, f64 z) {
        x = sycl::fmin(sycl::fmax(x * 1024., 0.), 1024. - 1.);
        y = sycl::fmin(sycl::fmax(y * 1024., 0.), 1024. - 1.);
        z = sycl::fmin(sycl::fmax(z * 1024., 0.), 1024. - 1.);

        u32 xx = shamrock::sfc::bmi::expand_bits<u32, 2>((u32)x);
        u32 yy = shamrock::sfc::bmi::expand_bits<u32, 2>((u32)y);
        u32 zz = shamrock::sfc::bmi::expand_bits<u32, 2>((u32)z);
        return xx * 4 + yy * 2 + zz;
    }

    template <> inline u32 coord_to_morton<u32, f32>(f32 x, f32 y, f32 z) {
        x = sycl::fmin(sycl::fmax(x * 1024.F, 0.F), 1024.F - 1.F);
        y = sycl::fmin(sycl::fmax(y * 1024.F, 0.F), 1024.F - 1.F);
        z = sycl::fmin(sycl::fmax(z * 1024.F, 0.F), 1024.F - 1.F);

        u32 xx = shamrock::sfc::bmi::expand_bits<u32, 2>((u32)x);
        u32 yy = shamrock::sfc::bmi::expand_bits<u32, 2>((u32)y);
        u32 zz = shamrock::sfc::bmi::expand_bits<u32, 2>((u32)z);
        return xx * 4 + yy * 2 + zz;
    }

    template <> inline u32_3 morton_to_ipos<u64>(u64 morton) {

        u32_3 pos;
        pos.x() = shamrock::sfc::bmi::contract_bits<u64, 2>((morton & 0x4924924924924924U) >> 2U);
        pos.y() = shamrock::sfc::bmi::contract_bits<u64, 2>((morton & 0x2492492492492492U) >> 1U);
        pos.z() = shamrock::sfc::bmi::contract_bits<u64, 2>((morton & 0x1249249249249249U) >> 0U);

        return pos;
    }

    template <> inline u16_3 morton_to_ipos<u32>(u32 morton) {

        u16_3 pos;
        pos.s0() = (u16)shamrock::sfc::bmi::contract_bits<u32, 2>((morton & 0x24924924U) >> 2U);
        pos.s1() = (u16)shamrock::sfc::bmi::contract_bits<u32, 2>((morton & 0x12492492U) >> 1U);
        pos.s2() = (u16)shamrock::sfc::bmi::contract_bits<u32, 2>((morton & 0x09249249U) >> 0U);

        return pos;
    }

    template <> inline u32_3 get_offset<u64>(uint clz_) {
        u32_3 mx;
        mx.s0() = 2097152U >> ((clz_ + 1) / 3);
        mx.s1() = 2097152U >> ((clz_ - 0) / 3);
        mx.s2() = 2097152U >> ((clz_ - 1) / 3);
        return mx;
    }

    template <> inline u16_3 get_offset<u32>(uint clz_) {
        u16_3 mx;
        mx.s0() = 1024U >> ((clz_ - 0) / 3);
        mx.s1() = 1024U >> ((clz_ - 1) / 3);
        mx.s2() = 1024U >> ((clz_ - 2) / 3);
        return mx;
    }

} // namespace morton_3d
