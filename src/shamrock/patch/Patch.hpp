// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Patch.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Header file for the patch struct and related function
 * @version 1.0
 * @date 2022-02-28
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "aliases.hpp"

#include "shamsys/MpiWrapper.hpp"

namespace shamrock::patch {

    template <u32 dim> MPI_Datatype get_patch_mpi_type();

    /**
     * @brief Patch object that contain generic patch information
     *
     */
    struct Patch {

        static constexpr u32 dim         = 3U;
        static constexpr u32 splts_count = 1U << dim;

        u64 id_patch; // unique key that identify the patch

        // load balancing fields

        u64 pack_node_index; ///< this value mean "to pack with index xxx in the global patch table"
                             ///< and not "to pack with id_pach == xxx"
        u64 load_value;      ///< if synchronized contain the load value of the patch

        // Data
        u64 x_min; ///< box coordinate of the corresponding patch
        u64 y_min; ///< box coordinate of the corresponding patch
        u64 z_min; ///< box coordinate of the corresponding patch
        u64 x_max; ///< box coordinate of the corresponding patch
        u64 y_max; ///< box coordinate of the corresponding patch
        u64 z_max; ///< box coordinate of the corresponding patch

        u32 data_count; ///< number of element in the corresponding patchdata

        u32 node_owner_id; ///< node rank owner of this patch

        /**
         * @brief check if patch equals
         *
         * @param rhs
         * @return true
         * @return false
         */
        inline bool operator==(const Patch &rhs) {

            bool ret_val = true;

            ret_val = ret_val && (id_patch == rhs.id_patch);

            ret_val = ret_val && (pack_node_index == rhs.pack_node_index);
            ret_val = ret_val && (load_value == rhs.load_value);

            ret_val = ret_val && (x_min == rhs.x_min);
            ret_val = ret_val && (y_min == rhs.y_min);
            ret_val = ret_val && (z_min == rhs.z_min);
            ret_val = ret_val && (x_max == rhs.x_max);
            ret_val = ret_val && (y_max == rhs.y_max);
            ret_val = ret_val && (z_max == rhs.z_max);
            ret_val = ret_val && (data_count == rhs.data_count);

            ret_val = ret_val && (node_owner_id == rhs.node_owner_id);

            return ret_val;
        }

        inline void set_err_mode() {
            // TODO notify in the documentation that this mean the patch is dead because it will be
            // flushed out when performing the sync
            node_owner_id = u32_max;
        }

        inline bool is_err_mode() { return node_owner_id == u32_max; }

        [[nodiscard]] std::array<u64, dim> get_split_coord() const;

        [[nodiscard]] std::array<Patch, splts_count> get_split() const;

        [[nodiscard]] static Patch merge_patch(std::array<Patch, splts_count> patches);

        /**
         * @brief return an open interval of the corresponding patch coordinates given the div & offset
         * 
         * @tparam T type to convert coordinates to
         * @param divfact 
         * @param offset 
         * @return std::tuple<sycl::vec<T,3>,sycl::vec<T,3>> 
         */
        template<class T> 
        std::tuple<sycl::vec<T,3>,sycl::vec<T,3>> convert_coord(
            sycl::vec<T,3> divfact, 
            sycl::vec<T,3> offset);

        /**
         * @brief check if particle is in the asked range, given the ouput of @convert_coord
         * 
         * @tparam T the type of coordinate
         * @param val the value to check against
         * @param min_val the min range given by convert_coord
         * @param max_val the max range given by convert_coord
         * @return true is in the patch
         * @return false is not in the patch
         */
        template<class T>
        inline static bool is_in_patch_converted(sycl::vec<T,3> val, sycl::vec<T,3> min_val, sycl::vec<T,3> max_val);

        

        
    };


    ////////////////////////////////////////////
    // out of line implementation
    ////////////////////////////////////////////

    template<class T> 
    inline std::tuple<sycl::vec<T,3>,sycl::vec<T,3>> Patch::convert_coord(
        sycl::vec<T,3> divfact, 
        sycl::vec<T,3> offset){

        using vec = sycl::vec<T,3>;

        vec min_bound = vec{x_min,y_min,z_min}/divfact + offset;
        vec max_bound = (vec{x_max,y_max,z_max}+ 1)/divfact + offset;

        return {min_bound, max_bound};
    }

    template<class T> 
    inline bool Patch::is_in_patch_converted(sycl::vec<T,3> val, sycl::vec<T,3> min_val, sycl::vec<T,3> max_val){
        return (
            (min_val.x() <= val.x()) && (val.x() < max_val.x()) &&
            (min_val.y() <= val.y()) && (val.y() < max_val.y()) &&
            (min_val.z() <= val.z()) && (val.z() < max_val.z()) 
        );
    }

    [[nodiscard]] inline auto Patch::get_split_coord() const -> std::array<u64, dim> {
        return {
            (((x_max - x_min) + 1) / 2) - 1 + x_min,
            (((y_max - y_min) + 1) / 2) - 1 + y_min,
            (((z_max - z_min) + 1) / 2) - 1 + z_min};
    }

    [[nodiscard]] inline auto Patch::get_split() const -> std::array<Patch, splts_count> {

        // init vars
        Patch p0, p1, p2, p3, p4, p5, p6, p7;

        // boundaries
        u64 min_x = x_min;
        u64 min_y = y_min;
        u64 min_z = z_min;

        u64 split_x = (((x_max - x_min) + 1) / 2) - 1 + min_x;
        u64 split_y = (((y_max - y_min) + 1) / 2) - 1 + min_y;
        u64 split_z = (((z_max - z_min) + 1) / 2) - 1 + min_z;

        u64 max_x = x_max;
        u64 max_y = y_max;
        u64 max_z = z_max;

        // setup internal fields
        p0 = *this; // copy of the current state
        p0.data_count /= 8;
        p0.load_value /= 8;

        p1 = p0;
        p2 = p0;
        p3 = p0;
        p4 = p0;
        p5 = p0;
        p6 = p0;
        p7 = p0;

        // apply bounds
        p0.x_min = min_x;
        p0.y_min = min_y;
        p0.z_min = min_z;
        p0.x_max = split_x;
        p0.y_max = split_y;
        p0.z_max = split_z;

        p1.x_min = min_x;
        p1.y_min = min_y;
        p1.z_min = split_z + 1;
        p1.x_max = split_x;
        p1.y_max = split_y;
        p1.z_max = max_z;

        p2.x_min = min_x;
        p2.y_min = split_y + 1;
        p2.z_min = min_z;
        p2.x_max = split_x;
        p2.y_max = max_y;
        p2.z_max = split_z;

        p3.x_min = min_x;
        p3.y_min = split_y + 1;
        p3.z_min = split_z + 1;
        p3.x_max = split_x;
        p3.y_max = max_y;
        p3.z_max = max_z;

        p4.x_min = split_x + 1;
        p4.y_min = min_y;
        p4.z_min = min_z;
        p4.x_max = max_x;
        p4.y_max = split_y;
        p4.z_max = split_z;

        p5.x_min = split_x + 1;
        p5.y_min = min_y;
        p5.z_min = split_z + 1;
        p5.x_max = max_x;
        p5.y_max = split_y;
        p5.z_max = max_z;

        p6.x_min = split_x + 1;
        p6.y_min = split_y + 1;
        p6.z_min = min_z;
        p6.x_max = max_x;
        p6.y_max = max_y;
        p6.z_max = split_z;

        p7.x_min = split_x + 1;
        p7.y_min = split_y + 1;
        p7.z_min = split_z + 1;
        p7.x_max = max_x;
        p7.y_max = max_y;
        p7.z_max = max_z;

        return {p0, p1, p2, p3, p4, p5, p6, p7};
    }

    [[nodiscard]] inline Patch Patch::merge_patch(std::array<Patch, splts_count> patches) {

        u64 min_x = patches[0].x_min;
        u64 min_y = patches[0].y_min;
        u64 min_z = patches[0].z_min;

        u64 max_x = patches[0].x_max;
        u64 max_y = patches[0].y_max;
        u64 max_z = patches[0].z_max;

        min_x = sycl::min(min_x, patches[1].x_min);
        min_y = sycl::min(min_y, patches[1].y_min);
        min_z = sycl::min(min_z, patches[1].z_min);
        max_x = sycl::max(max_x, patches[1].x_max);
        max_y = sycl::max(max_y, patches[1].y_max);
        max_z = sycl::max(max_z, patches[1].z_max);

        min_x = sycl::min(min_x, patches[2].x_min);
        min_y = sycl::min(min_y, patches[2].y_min);
        min_z = sycl::min(min_z, patches[2].z_min);
        max_x = sycl::max(max_x, patches[2].x_max);
        max_y = sycl::max(max_y, patches[2].y_max);
        max_z = sycl::max(max_z, patches[2].z_max);

        min_x = sycl::min(min_x, patches[3].x_min);
        min_y = sycl::min(min_y, patches[3].y_min);
        min_z = sycl::min(min_z, patches[3].z_min);
        max_x = sycl::max(max_x, patches[3].x_max);
        max_y = sycl::max(max_y, patches[3].y_max);
        max_z = sycl::max(max_z, patches[3].z_max);

        min_x = sycl::min(min_x, patches[4].x_min);
        min_y = sycl::min(min_y, patches[4].y_min);
        min_z = sycl::min(min_z, patches[4].z_min);
        max_x = sycl::max(max_x, patches[4].x_max);
        max_y = sycl::max(max_y, patches[4].y_max);
        max_z = sycl::max(max_z, patches[4].z_max);

        min_x = sycl::min(min_x, patches[5].x_min);
        min_y = sycl::min(min_y, patches[5].y_min);
        min_z = sycl::min(min_z, patches[5].z_min);
        max_x = sycl::max(max_x, patches[5].x_max);
        max_y = sycl::max(max_y, patches[5].y_max);
        max_z = sycl::max(max_z, patches[5].z_max);

        min_x = sycl::min(min_x, patches[6].x_min);
        min_y = sycl::min(min_y, patches[6].y_min);
        min_z = sycl::min(min_z, patches[6].z_min);
        max_x = sycl::max(max_x, patches[6].x_max);
        max_y = sycl::max(max_y, patches[6].y_max);
        max_z = sycl::max(max_z, patches[6].z_max);

        min_x = sycl::min(min_x, patches[7].x_min);
        min_y = sycl::min(min_y, patches[7].y_min);
        min_z = sycl::min(min_z, patches[7].z_min);
        max_x = sycl::max(max_x, patches[7].x_max);
        max_y = sycl::max(max_y, patches[7].y_max);
        max_z = sycl::max(max_z, patches[7].z_max);

        Patch ret{};
        ret = patches[0];

        ret.x_min = min_x;
        ret.y_min = min_y;
        ret.z_min = min_z;
        ret.x_max = max_x;
        ret.y_max = max_y;
        ret.z_max = max_z;

        ret.pack_node_index = u64_max;

        ret.load_value += patches[1].load_value;
        ret.load_value += patches[2].load_value;
        ret.load_value += patches[3].load_value;
        ret.load_value += patches[4].load_value;
        ret.load_value += patches[5].load_value;
        ret.load_value += patches[6].load_value;
        ret.load_value += patches[7].load_value;

        ret.data_count += patches[1].data_count;
        ret.data_count += patches[2].data_count;
        ret.data_count += patches[3].data_count;
        ret.data_count += patches[4].data_count;
        ret.data_count += patches[5].data_count;
        ret.data_count += patches[6].data_count;
        ret.data_count += patches[7].data_count;

        return ret;
    }

} // namespace shamrock::patch