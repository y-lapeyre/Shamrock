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
#include "PatchCoord.hpp"

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

        [[nodiscard]] inline bool is_err_mode() const { return node_owner_id == u32_max; }

        [[nodiscard]] std::array<u64, dim> get_split_coord() const;

        [[nodiscard]] std::array<Patch, splts_count> get_split() const;

        [[nodiscard]] static Patch merge_patch(std::array<Patch, splts_count> patches);

        /**
         * @brief return an open interval of the corresponding patch coordinates given the div &
         * offset
         *
         * @tparam T type to convert coordinates to
         * @param divfact
         * @param offset
         * @return std::tuple<sycl::vec<T,3>,sycl::vec<T,3>>
         */
        template <class T>
        std::tuple<sycl::vec<T, 3>, sycl::vec<T, 3>>
        convert_coord(sycl::vec<u64, 3> src_offset, sycl::vec<T, 3> divfact, sycl::vec<T, 3> offset) const ;

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
        template <class T>
        inline static bool is_in_patch_converted(
            sycl::vec<T, 3> val, sycl::vec<T, 3> min_val, sycl::vec<T, 3> max_val
        );

        inline void override_from_coord(PatchCoord pc) {
            x_min = pc.x_min;
            y_min = pc.y_min;
            z_min = pc.z_min;
            x_max = pc.x_max;
            y_max = pc.y_max;
            z_max = pc.z_max;
        }

        [[nodiscard]] inline PatchCoord get_coords() const {
            return PatchCoord(x_min, y_min, z_min, x_max, y_max, z_max);
        }
    };

    ////////////////////////////////////////////
    // out of line implementation
    ////////////////////////////////////////////

    template <class T>
    inline std::tuple<sycl::vec<T, 3>, sycl::vec<T, 3>>
    Patch::convert_coord(sycl::vec<u64, 3> src_offset, sycl::vec<T, 3> divfact, sycl::vec<T, 3> offset) const {
        return PatchCoord::convert_coord( x_min,  y_min,  z_min,  x_max,  y_max,  z_max,
        src_offset.x(),src_offset.y(),src_offset.z()
        ,  divfact,  offset);
    }

    template <class T>
    inline bool Patch::is_in_patch_converted(
        sycl::vec<T, 3> val, sycl::vec<T, 3> min_val, sycl::vec<T, 3> max_val
    ) {
        return (
            (min_val.x() <= val.x()) && (val.x() < max_val.x()) && (min_val.y() <= val.y()) &&
            (val.y() < max_val.y()) && (min_val.z() <= val.z()) && (val.z() < max_val.z())
        );
    }

    [[nodiscard]] inline auto Patch::get_split_coord() const -> std::array<u64, dim> {
        return PatchCoord::get_split_coord(x_min, y_min, z_min, x_max, y_max, z_max);
    }

    [[nodiscard]] inline auto Patch::get_split() const -> std::array<Patch, splts_count> {

        // init vars
        Patch p0, p1, p2, p3, p4, p5, p6, p7;

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

        std::array<PatchCoord, splts_count> splts_c =
            PatchCoord::get_split(x_min, y_min, z_min, x_max, y_max, z_max);

        p0.override_from_coord(splts_c[0]);
        p1.override_from_coord(splts_c[1]);
        p2.override_from_coord(splts_c[2]);
        p3.override_from_coord(splts_c[3]);
        p4.override_from_coord(splts_c[4]);
        p5.override_from_coord(splts_c[5]);
        p6.override_from_coord(splts_c[6]);
        p7.override_from_coord(splts_c[7]);

        return {p0, p1, p2, p3, p4, p5, p6, p7};
    }

    [[nodiscard]] inline Patch Patch::merge_patch(std::array<Patch, splts_count> patches) {

        PatchCoord merged_c = PatchCoord::merge(
            {patches[0].get_coords(),
             patches[1].get_coords(),
             patches[2].get_coords(),
             patches[3].get_coords(),
             patches[4].get_coords(),
             patches[5].get_coords(),
             patches[6].get_coords(),
             patches[7].get_coords()}
        );

        Patch ret{};
        ret = patches[0];

        ret.x_min = merged_c.x_min;
        ret.y_min = merged_c.y_min;
        ret.z_min = merged_c.z_min;
        ret.x_max = merged_c.x_max;
        ret.y_max = merged_c.y_max;
        ret.z_max = merged_c.z_max;

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