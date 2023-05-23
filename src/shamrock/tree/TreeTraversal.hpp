// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl.hpp"
#include "shamrock/tree/RadixTree.hpp"

namespace shamrock::tree {

    template<class u_morton, class vec>
    class ObjectIterator {

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> particle_index_map;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> cell_index_map;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> rchild_id;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> lchild_id;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> rchild_flag;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> lchild_flag;
        sycl::accessor<vec, 1, sycl::access::mode::read, sycl::target::device> pos_min_cell;
        sycl::accessor<vec, 1, sycl::access::mode::read, sycl::target::device> pos_max_cell;

        static constexpr u32 tree_depth = RadixTree<u_morton, vec, 3>::tree_depth;
        static constexpr u32 _nindex    = 4294967295;

        u32 leaf_offset;

        public:

        // clang-format off
        ObjectIterator(RadixTree< u_morton,  vec,3> & rtree,sycl::handler & cgh):
            particle_index_map{shambase::get_check_ref(rtree.tree_morton_codes.buf_particle_index_map), cgh,sycl::read_only},
            cell_index_map{shambase::get_check_ref(rtree.tree_reduced_morton_codes.buf_reduc_index_map), cgh,sycl::read_only},
            rchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_rchild_id)  , cgh,sycl::read_only},
            lchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_lchild_id)  , cgh,sycl::read_only},
            rchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_rchild_flag), cgh,sycl::read_only},
            lchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_lchild_flag), cgh,sycl::read_only},
            pos_min_cell  {shambase::get_check_ref(rtree.tree_cell_ranges.buf_pos_min_cell_flt), cgh,sycl::read_only},
            pos_max_cell  {shambase::get_check_ref(rtree.tree_cell_ranges.buf_pos_max_cell_flt), cgh,sycl::read_only},
            leaf_offset   (rtree.tree_struct.internal_cell_count)
        {}
        // clang-format on

        template<class Functor_iter>
        inline void iter_object_in_cell(const u32 & cell_id, Functor_iter &&func_it) const {
            // loop on particle indexes
            uint min_ids = cell_index_map[cell_id     -leaf_offset];
            uint max_ids = cell_index_map[cell_id + 1 -leaf_offset];

            for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                //recover old index before morton sort
                uint id_b = particle_index_map[id_s];

                //iteration function
                func_it(id_b);

            }
            
        }

        template<class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
        inline void rtree_for(Functor_int_cd &&func_int_cd,
                                   Functor_iter &&func_it,
                                   Functor_iter_excl &&func_excl) const {
            u32 stack_cursor = tree_depth - 1;
            std::array<u32, tree_depth> id_stack;
            id_stack[stack_cursor] = 0;

            while (stack_cursor < tree_depth) {

                u32 current_node_id    = id_stack[stack_cursor];
                id_stack[stack_cursor] = _nindex;
                stack_cursor++;

                bool cur_id_valid = func_int_cd(
                    current_node_id, pos_min_cell[current_node_id], pos_max_cell[current_node_id]);

                if (cur_id_valid) {

                    // leaf and cell can interact
                    if (current_node_id >= leaf_offset) {

                        iter_object_in_cell(current_node_id, func_it);

                        // can interact not leaf => stack
                    } else {

                        u32 lid =
                            lchild_id[current_node_id] + leaf_offset * lchild_flag[current_node_id];
                        u32 rid =
                            rchild_id[current_node_id] + leaf_offset * rchild_flag[current_node_id];

                        id_stack[stack_cursor - 1] = rid;
                        stack_cursor--;

                        id_stack[stack_cursor - 1] = lid;
                        stack_cursor--;
                    }
                } else {
                    // grav
                    func_excl(current_node_id);
                }
            }
        }

        template<class Functor_int_cd, class Functor_iter>
        inline void rtree_for(Functor_int_cd &&func_int_cd, Functor_iter &&func_it) const {
            rtree_for(std::forward<Functor_int_cd>(func_int_cd),
                           std::forward<Functor_iter>(func_it),
                           [](u32) {});
        }
    };




    template<class u_morton, class vec>
    class LeafIterator {

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> rchild_id;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> lchild_id;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> rchild_flag;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> lchild_flag;

        public:
        sycl::accessor<vec, 1, sycl::access::mode::read, sycl::target::device> pos_min_cell;
        sycl::accessor<vec, 1, sycl::access::mode::read, sycl::target::device> pos_max_cell;
        private:

        static constexpr u32 tree_depth = RadixTree<u_morton, vec, 3>::tree_depth;
        static constexpr u32 _nindex    = 4294967295;

        u32 leaf_offset;

        public:

        // clang-format off
        LeafIterator(RadixTree< u_morton,  vec,3> & rtree,sycl::handler & cgh):
            rchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_rchild_id)  , cgh,sycl::read_only},
            lchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_lchild_id)  , cgh,sycl::read_only},
            rchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_rchild_flag), cgh,sycl::read_only},
            lchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_lchild_flag), cgh,sycl::read_only},
            pos_min_cell  {shambase::get_check_ref(rtree.tree_cell_ranges.buf_pos_min_cell_flt), cgh,sycl::read_only},
            pos_max_cell  {shambase::get_check_ref(rtree.tree_cell_ranges.buf_pos_max_cell_flt), cgh,sycl::read_only},
            leaf_offset   (rtree.tree_struct.internal_cell_count)
        {}
        // clang-format on

        template<class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
        inline void rtree_for(Functor_int_cd &&func_int_cd,
                                   Functor_iter &&func_it,
                                   Functor_iter_excl &&func_excl) const {
            u32 stack_cursor = tree_depth - 1;
            std::array<u32, tree_depth> id_stack;
            id_stack[stack_cursor] = 0;

            while (stack_cursor < tree_depth) {

                u32 current_node_id    = id_stack[stack_cursor];
                id_stack[stack_cursor] = _nindex;
                stack_cursor++;

                bool cur_id_valid = func_int_cd(
                    current_node_id, pos_min_cell[current_node_id], pos_max_cell[current_node_id]);

                if (cur_id_valid) {

                    // leaf and cell can interact
                    if (current_node_id >= leaf_offset) {

                        func_it(current_node_id);

                        // can interact not leaf => stack
                    } else {

                        u32 lid =
                            lchild_id[current_node_id] + leaf_offset * lchild_flag[current_node_id];
                        u32 rid =
                            rchild_id[current_node_id] + leaf_offset * rchild_flag[current_node_id];

                        id_stack[stack_cursor - 1] = rid;
                        stack_cursor--;

                        id_stack[stack_cursor - 1] = lid;
                        stack_cursor--;
                    }
                } else {
                    // grav
                    func_excl(current_node_id);
                }
            }
        }

        template<class Functor_int_cd, class Functor_iter>
        inline void rtree_for(Functor_int_cd &&func_int_cd, Functor_iter &&func_it) const {
            rtree_for(std::forward<Functor_int_cd>(func_int_cd),
                           std::forward<Functor_iter>(func_it),
                           [](u32) {});
        }
    };


    


} // namespace shamrock::tree