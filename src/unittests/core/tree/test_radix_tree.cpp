#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "unittests/shamrocktest.hpp"


#include "core/tree/radix_tree.hpp"
#include <vector>


Test_start("radix_tree", tree_cut, 1){

    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 400;

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
        for (u32 i = 0; i < rtree.tree_internal_count+ rtree.tree_leaf_count; i++) {
            acc[i] = i;
        }
    }


    {
        logger::debug_sycl_ln("Radixtree", "valid_node_state");
        rtree.print_tree_field(node_id_old);
        logger::raw_ln("");
    }


    auto ret = rtree.cut_tree(sycl_handler::get_compute_queue(), {vec{-1,-1,-1},vec{-0.0001,1,1}}, pdat);

}