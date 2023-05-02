// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "kernels/compute_ranges.hpp"
#include "kernels/convert_ranges.hpp"
#include "shamalgs/memory/memory.hpp"
#include "shambase/sycl.hpp"
#include "shamrock/sfc/MortonKernels.hpp"
#include "shamrock/sfc/morton.hpp"
#include "shamrock/tree/TreeReducedMortonCodes.hpp"
#include "shamrock/tree/TreeStructure.hpp"

namespace shamrock::tree {

    template<class u_morton, class pos_t>
    class TreeCellRanges {

        using Morton             = shamrock::sfc::MortonCodes<u_morton, 3>;
        using ipos_t             = typename shamrock::sfc::MortonCodes<u_morton, 3>::int_vec_repr;
        static constexpr u32 dim = 3;

        public:
        // this one is not used, it should be removed
        std::unique_ptr<sycl::buffer<ipos_t>>
            buf_pos_min_cell; // size = total count //rename to ipos
        std::unique_ptr<sycl::buffer<ipos_t>> buf_pos_max_cell; // size = total count

        // optional
        std::unique_ptr<sycl::buffer<pos_t>>
            buf_pos_min_cell_flt; // size = total count //drop the flt part
        std::unique_ptr<sycl::buffer<pos_t>> buf_pos_max_cell_flt; // size = total count

        inline void build1(sycl::queue &queue,
                           TreeReducedMortonCodes<u_morton> &tree_reduced_morton_codes,
                           TreeStructure<u_morton> &tree_struct) {
            if (!tree_struct.one_cell_mode) {

                logger::debug_sycl_ln("RadixTree", "compute_cellvolume");

                buf_pos_min_cell = std::make_unique<sycl::buffer<ipos_t>>(
                    tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);
                buf_pos_max_cell = std::make_unique<sycl::buffer<ipos_t>>(
                    tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);

                sycl_compute_cell_ranges(queue,
                                         tree_reduced_morton_codes.tree_leaf_count,
                                         tree_struct.internal_cell_count,
                                         tree_reduced_morton_codes.buf_tree_morton,
                                         tree_struct.buf_lchild_id,
                                         tree_struct.buf_rchild_id,
                                         tree_struct.buf_lchild_flag,
                                         tree_struct.buf_rchild_flag,
                                         tree_struct.buf_endrange,
                                         buf_pos_min_cell,
                                         buf_pos_max_cell);

            } else {
                // throw shamrock_exc("one cell mode is not implemented");
                // TODO do some extensive test on one cell mode

                buf_pos_min_cell = std::make_unique<sycl::buffer<ipos_t>>(
                    tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);
                buf_pos_max_cell = std::make_unique<sycl::buffer<ipos_t>>(
                    tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);

                {

                    sycl::host_accessor pos_min_cell{
                        *buf_pos_min_cell, sycl::write_only, sycl::no_init};
                    sycl::host_accessor pos_max_cell{
                        *buf_pos_max_cell, sycl::write_only, sycl::no_init};

                    pos_min_cell[0] = {0, 0, 0};
                    pos_max_cell[0] = {Morton::max_val, Morton::max_val, Morton::max_val};

                    pos_min_cell[1] = {0, 0, 0};
                    pos_max_cell[1] = {Morton::max_val, Morton::max_val, Morton::max_val};

                    pos_min_cell[2] = {0, 0, 0};
                    pos_max_cell[2] = {0, 0, 0};

                    logger::debug_sycl_ln("RadixTree", "compute_cellvolume one cell mode");
                    logger::debug_sycl_ln("RadixTree",
                                          " -> ",
                                          pos_min_cell[0],
                                          pos_max_cell[0],
                                          pos_min_cell[1],
                                          pos_max_cell[1],
                                          pos_min_cell[2],
                                          pos_max_cell[2],
                                          "len =",
                                          tree_struct.internal_cell_count +
                                              tree_reduced_morton_codes.tree_leaf_count);
                }
            }
        }

        void build2(sycl::queue &queue, u32 total_count, std::tuple<pos_t, pos_t> bounding_box) {

            buf_pos_min_cell_flt = std::make_unique<sycl::buffer<pos_t>>(total_count);
            buf_pos_max_cell_flt = std::make_unique<sycl::buffer<pos_t>>(total_count);

            logger::debug_sycl_ln("RadixTree", "sycl_convert_cell_range");

            shamrock::sfc::MortonKernels<u_morton, pos_t, dim>::sycl_irange_to_range(
                queue,
                total_count,
                std::get<0>(bounding_box),
                std::get<1>(bounding_box),
                buf_pos_min_cell,
                buf_pos_max_cell,
                buf_pos_min_cell_flt,
                buf_pos_max_cell_flt);

            //remove old buf ?
        }

        inline bool are_range_int_built() {
            return bool(buf_pos_min_cell) && bool(buf_pos_max_cell);
        }

        inline bool are_range_float_built() {
            return bool(buf_pos_min_cell_flt) && bool(buf_pos_max_cell_flt);
        }

        inline TreeCellRanges() = default;

        inline TreeCellRanges(const TreeCellRanges &other)
            : buf_pos_min_cell(
                  shamalgs::memory::duplicate(other.buf_pos_min_cell)), // size = total count
              buf_pos_max_cell(
                  shamalgs::memory::duplicate(other.buf_pos_max_cell)), // size = total count
              buf_pos_min_cell_flt(
                  shamalgs::memory::duplicate(other.buf_pos_min_cell_flt)), // size = total count
              buf_pos_max_cell_flt(
                  shamalgs::memory::duplicate(other.buf_pos_max_cell_flt)) // size = total count
        {}

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

            add_ptr(buf_pos_min_cell);
            add_ptr(buf_pos_max_cell);
            add_ptr(buf_pos_min_cell_flt);
            add_ptr(buf_pos_max_cell_flt);

            return sum;
        }
    };

} // namespace shamrock::tree