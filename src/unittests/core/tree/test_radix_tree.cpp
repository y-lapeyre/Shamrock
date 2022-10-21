#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "unittests/shamrocktest.hpp"


#include "core/tree/radix_tree.hpp"
#include <vector>





Test_start("radix_tree",inclusion_ok,1){
    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;
    using vec3i = morton_3d::morton_types<morton_mode>::int_vec_repr;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 1000;

    PatchDataLayout pdl;
    pdl.add_field<vec>("xyz", 1);

    const auto id_xyz = pdl.get_field_idx<vec>("xyz");

    PatchData pdat(pdl);
    pdat.resize(npart);

    

    {
        auto & pos_part = pdat.get_field<vec>(id_xyz).get_buf();
        sycl::host_accessor<vec> pos {*pos_part};

        for (u32 i = 0; i < npart; i ++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
        }
    }

    auto rtree = Radix_Tree<morton_mode, vec>(
            sycl_handler::get_compute_queue(), 
            {vec{-1,-1,-1},vec{1,1,1}},
            pdat.get_field<vec>(id_xyz).get_buf(), 
            npart 
        );

    rtree.compute_cellvolume(sycl_handler::get_compute_queue());

    {
        sycl::host_accessor tree_acc_pos_min_cell{*rtree.buf_pos_min_cell,sycl::read_only};
        sycl::host_accessor tree_acc_pos_max_cell{*rtree.buf_pos_max_cell,sycl::read_only};
        u32 tree_leaf_offset = rtree.tree_internal_count;
        sycl::host_accessor tree_lchild_id   {*rtree.buf_lchild_id  ,sycl::read_only};
        sycl::host_accessor tree_rchild_id   {*rtree.buf_rchild_id  ,sycl::read_only};
        sycl::host_accessor tree_lchild_flag {*rtree.buf_lchild_flag,sycl::read_only};
        sycl::host_accessor tree_rchild_flag {*rtree.buf_rchild_flag,sycl::read_only};

        for (u32 i = 0; i< rtree.tree_internal_count; i++) {
            vec3i cur_pos_min_cell_a = tree_acc_pos_min_cell[i];
            vec3i cur_pos_max_cell_a = tree_acc_pos_max_cell[i];

            auto inclusion_crit = [&](vec3i other_min, vec3i other_max) -> bool {
                return 
                    (cur_pos_min_cell_a.x() <= other_min.x()) && 
                    (cur_pos_min_cell_a.y() <= other_min.y()) && 
                    (cur_pos_min_cell_a.z() <= other_min.z()) && 
                    (cur_pos_max_cell_a.x() >= other_max.x()) && 
                    (cur_pos_max_cell_a.y() >= other_max.y()) && 
                    (cur_pos_max_cell_a.z() >= other_max.z()) ; 
            };


            u32 lid = tree_lchild_id[i] + tree_leaf_offset * tree_lchild_flag[i];
            u32 rid = tree_rchild_id[i] + tree_leaf_offset * tree_rchild_flag[i];

            vec3i cur_pos_min_cell_bl = tree_acc_pos_min_cell[lid];
            vec3i cur_pos_max_cell_bl = tree_acc_pos_max_cell[lid];

            vec3i cur_pos_min_cell_br = tree_acc_pos_min_cell[rid];
            vec3i cur_pos_max_cell_br = tree_acc_pos_max_cell[rid];

            bool l_ok = inclusion_crit(cur_pos_min_cell_bl,cur_pos_max_cell_bl);
            bool r_ok = inclusion_crit(cur_pos_min_cell_br,cur_pos_max_cell_br);

            Test_assert("inclusion", l_ok && r_ok);

        }
    }
    

}

















Test_start("radix_tree", tree_cut, 1){

    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 1000;

    PatchDataLayout pdl;
    pdl.add_field<vec>("xyz", 1);

    const auto id_xyz = pdl.get_field_idx<vec>("xyz");

    PatchData pdat(pdl);
    pdat.resize(npart);

    

    {
        auto & pos_part = pdat.get_field<vec>(id_xyz).get_buf();
        sycl::host_accessor<vec> pos {*pos_part};

        for (u32 i = 0; i < npart; i ++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
        }
    }

    auto rtree = Radix_Tree<morton_mode, vec>(
            sycl_handler::get_compute_queue(), 
            {vec{-1,-1,-1},vec{1,1,1}},
            pdat.get_field<vec>(id_xyz).get_buf(), 
            npart 
        );

    rtree.compute_cellvolume(sycl_handler::get_compute_queue());




    sycl::buffer<u32> node_id_old = sycl::buffer<u32>(rtree.tree_internal_count+ rtree.tree_leaf_count);
    {
        sycl::host_accessor acc {node_id_old};
        for (u32 i = 0; i < rtree.tree_internal_count; i++) {
            acc[i] = i;
        }

        for (u32 i = 0; i < rtree.tree_leaf_count; i++) {
            acc[i + rtree.tree_internal_count] = i;
        }
    }


    {
        logger::debug_sycl_ln("Radixtree", "valid_node_state");
        rtree.print_tree_field(node_id_old);
        logger::raw_ln("");
    }


    auto [new_tree,remap_buf_field,new_pdat] = rtree.cut_tree(sycl_handler::get_compute_queue(), {vec{-1,-1,-1},vec{0,0.5,1}}, pdat);



}