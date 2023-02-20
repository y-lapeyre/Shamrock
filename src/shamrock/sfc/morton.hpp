// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

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

#include "aliases.hpp"
#include "bmi.hpp"
#include "shamutils/sycl_utils/vectorProperties.hpp"
#include "shamrock/math/integerManip.hpp"
#include <type_traits>


namespace shamrock::sfc {

        
    template<class morton_repr, u32 dim>
    class MortonCodes{};



    template<> class MortonCodes<u32, 3>{public:

        using int_vec_repr_base                    = u16;
        using int_vec_repr                         = u16_3;
        static constexpr int_vec_repr_base max_val = 1024 - 1;
        static constexpr int_vec_repr_base val_count = 1024;

        static constexpr u32 err_code = 4294967295U;

        inline static u32 icoord_to_morton(u32 x, u32 y, u32 z){
            u32 xx = bmi::expand_bits<u32, 2>((u32)x);
            u32 yy = bmi::expand_bits<u32, 2>((u32)y);
            u32 zz = bmi::expand_bits<u32, 2>((u32)z);
            return xx * 4 + yy * 2 + zz;
        }

        inline static bool is_morton_bounding_box(int_vec_repr min, int_vec_repr max) noexcept{
            return  
                min.x() == 0 &&
                min.y() == 0 &&
                min.z() == 0 &&
                max.x() == max_val &&
                max.y() == max_val &&
                max.z() == max_val
            ;
        }

        template<class flt>
        inline static u32 coord_to_morton(flt x, flt y, flt z){

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
        static constexpr int_vec_repr_base val_count = 2097152;

        static constexpr u64 err_code = 18446744073709551615UL;

        inline static u64 icoord_to_morton(u64 x, u64 y, u64 z){
            u64 xx = bmi::expand_bits<u64, 2>((u64)x);
            u64 yy = bmi::expand_bits<u64, 2>((u64)y);
            u64 zz = bmi::expand_bits<u64, 2>((u64)z);
            return xx * 4 + yy * 2 + zz;
        }


        inline static bool is_morton_bounding_box(int_vec_repr min, int_vec_repr max) noexcept{
            return  
                min.x() == 0 &&
                min.y() == 0 &&
                min.z() == 0 &&
                max.x() == max_val &&
                max.y() == max_val &&
                max.z() == max_val
            ;
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



    template<class morton_t,class _pos_t, u32 dim>
    class MortonConverter{
        
        
        
        public:

        using Morton = MortonCodes<morton_t, dim>;

        using pos_t = _pos_t;
        using coord_t = typename shamutils::sycl_utils::VectorProperties<pos_t>::component_type;
        using ipos_t  = typename Morton::int_vec_repr;
        using int_t = typename Morton::int_vec_repr_base;


        private : 

        static constexpr bool implemented_int = 
            std::is_same<pos_t, u32_3>::value ||
            std::is_same<pos_t, u64_3>::value;

        static constexpr bool implemented_float = 
            std::is_same<pos_t, f32_3>::value ||
            std::is_same<pos_t, f64_3>::value;

        
        static_assert(implemented_int || implemented_float, "not implemented");

        public:

        static std::tuple<pos_t,pos_t> get_transform(pos_t bounding_box_min,pos_t bounding_box_max);




        inline static ipos_t to_morton_grid(pos_t pos, pos_t origin, pos_t scale){
            
            pos_t unit_coord = (pos - origin)/scale;

            constexpr coord_t max_bound = Morton::max_val;
            constexpr coord_t zero = 0;
            
            
            if constexpr (implemented_float){
                unit_coord.x() = sycl::fmin(sycl::fmax(unit_coord.x() , zero), max_bound);
                unit_coord.y() = sycl::fmin(sycl::fmax(unit_coord.y() , zero), max_bound);
                unit_coord.z() = sycl::fmin(sycl::fmax(unit_coord.z() , zero), max_bound);
            }

            if constexpr (implemented_int){
                unit_coord.x() = sycl::min(sycl::max(unit_coord.x() , zero), max_bound);
                unit_coord.y() = sycl::min(sycl::max(unit_coord.y() , zero), max_bound);
                unit_coord.z() = sycl::min(sycl::max(unit_coord.z() , zero), max_bound);
            }

            return ipos_t{int_t(unit_coord.x()),int_t(unit_coord.y()),int_t(unit_coord.z())};
        }




        inline static pos_t to_real_space(ipos_t pos, pos_t origin, pos_t scale){
            pos_t r;
            
            constexpr bool implemented_type = dim == 3;
            static_assert(implemented_type, "the required case is not implemented");

            if constexpr (dim == 3){
                r = pos_t{pos.x(), pos.y(), pos.z()};
            }

            r *= scale;
            r += origin;
            return r;
        }

    };

    template<class morton_t,class _pos_t, u32 dim>
    inline auto MortonConverter<morton_t,_pos_t,dim>::get_transform(pos_t bounding_box_min, pos_t bounding_box_max) -> std::tuple<pos_t,pos_t>{
        if constexpr (implemented_float){
            constexpr coord_t grd_scale = Morton::val_count;
            return {bounding_box_min, (bounding_box_max - bounding_box_min)/grd_scale};
        }

        if constexpr (implemented_int){
            // to convert int coord to morton we need to have a cubic box
            // with 2^n side lenght >= max morton side lenght

            coord_t dx = bounding_box_max.x() - bounding_box_min.x();
            coord_t dy = bounding_box_max.y() - bounding_box_min.y();
            coord_t dz = bounding_box_max.z() - bounding_box_min.z();

            if (!(dx == dy && dy == dz)) {
                throw std::invalid_argument("The bounding box is ot a cube");
            }

            // validity condition
            auto check_axis = [](coord_t len) -> bool {

                if (len % Morton::val_count == 0) {
                    return true;
                }
                return false;
            };

            if (!check_axis(dx)) {
                throw excep_with_pos(std::invalid_argument,
                    "The x axis bounding box is not a pow of 2 with lenght >= morton_max\n"+
                    std::to_string(dx) + " " + std::to_string(dx % Morton::val_count)
                );
            }

            if (!check_axis(dy)) {
                throw excep_with_pos(std::invalid_argument,
                    "The y axis bounding box is not a pow of 2 with lenght >= morton_max\n"+
                    std::to_string(dy) + " " + std::to_string(dy % Morton::val_count)
                );
            }

            if (!check_axis(dz)) {
                throw excep_with_pos(std::invalid_argument,
                    "The z axis bounding box is not a pow of 2 with lenght >= morton_max\n"+
                    std::to_string(dz) + " " + std::to_string(dz % Morton::val_count)
                );
            }

            // we can only use dx since they must be equal to reach this call
            coord_t scale = dx / (Morton::val_count);

            return {bounding_box_min,pos_t{scale,scale,scale}};
        }
    }



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
