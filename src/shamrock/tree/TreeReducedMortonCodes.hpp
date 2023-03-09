// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shammath/CoordRange.hpp"
#include "shamrock/legacy/algs/sycl/defs.hpp"
#include "shamrock/tree/RadixTreeMortonBuilder.hpp"
#include "shamrock/tree/TreeMortonCodes.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamutils/stringUtils.hpp"

#include "kernels/reduction_alg.hpp"

namespace shamrock::tree {

    template<class u_morton>
    class TreeReducedMortonCodes {
        public:
        u32 tree_leaf_count;
        std::unique_ptr<sycl::buffer<u32>> buf_reduc_index_map;
        std::unique_ptr<sycl::buffer<u_morton>> buf_tree_morton; // size = leaf cnt

        inline void build(
            sycl::queue &queue,
            u32 obj_cnt,
            u32 reduc_level,
            TreeMortonCodes<u_morton> &morton_codes,

            bool &one_cell_mode
        ) {

            // return a sycl buffer from reduc index map instead
            logger::debug_sycl_ln(
                "RadixTree", "reduction algorithm"
            ); // TODO put reduction level in class member
            std::vector<u32> reduc_index_map;
            reduction_alg(
                queue,
                obj_cnt,
                morton_codes.buf_morton,
                reduc_level,
                reduc_index_map,
                tree_leaf_count
            );

            logger::debug_sycl_ln(
                "RadixTree",
                "reduction results : (before :",
                obj_cnt,
                " | after :",
                tree_leaf_count,
                ") ratio :",
                shamutils::format_printf("%2.2f", f32(obj_cnt) / f32(tree_leaf_count))
            );

            if (tree_leaf_count > 1) {

                // buf_reduc_index_map =
                // std::make_unique<sycl::buffer<u32>>(reduc_index_map.data(),reduc_index_map.size());

                buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(
                    syclalgs::convert::vector_to_buf(reduc_index_map)
                );

                logger::debug_sycl_ln("RadixTree", "sycl_morton_remap_reduction");
                buf_tree_morton = std::make_unique<sycl::buffer<u_morton>>(tree_leaf_count);

                sycl_morton_remap_reduction(
                    queue,
                    tree_leaf_count,
                    buf_reduc_index_map,
                    morton_codes.buf_morton,
                    buf_tree_morton
                );

                one_cell_mode = false;

            } else if (tree_leaf_count == 1) {
                // throw shamrock_exc("one cell mode is not implemented");
                // TODO do some extensive test on one cell mode

                tree_leaf_count = 2;
                //reduc_index_map.push_back(-1);

                buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(
                    syclalgs::convert::vector_to_buf(reduc_index_map)
                );

                one_cell_mode = true;

            } else {
                throw shamutils::throw_with_loc<std::runtime_error>("empty patch should be skipped"
                );
            }
        }

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;

            sum += sizeof(tree_leaf_count);

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

            add_ptr(buf_reduc_index_map);
            add_ptr(buf_tree_morton);

            return sum;
        }

        inline TreeReducedMortonCodes() = default;

        inline TreeReducedMortonCodes(const TreeReducedMortonCodes &other)
            : tree_leaf_count(other.tree_leaf_count),
              buf_reduc_index_map(syclalgs::basic::duplicate(other.buf_reduc_index_map)),
              buf_tree_morton(syclalgs::basic::duplicate(other.buf_tree_morton)) {}


    };

} // namespace shamrock::tree