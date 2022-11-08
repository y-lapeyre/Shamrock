#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "unittests/shamrocktest.hpp"
#include "unittests/shamrockbench.hpp"

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

    
    constexpr u32 reduc_level = 5;

    auto rtree = Radix_Tree<morton_mode, vec>(
            sycl_handler::get_compute_queue(), 
            {vec{-1,-1,-1},vec{1,1,1}},
            pdat.get_field<vec>(id_xyz).get_buf(), 
            npart , reduc_level
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







Test_start("radix_tree",test_new_pfield_compute,1){
    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;
    using vec3i = morton_3d::morton_types<morton_mode>::int_vec_repr;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 1000;

    PatchDataLayout pdl;
    pdl.add_field<vec>("xyz", 1);
    pdl.add_field<flt>("h", 1);

    const auto id_xyz = pdl.get_field_idx<vec>("xyz");
    const auto id_h = pdl.get_field_idx<flt>("h");

    PatchData pdat(pdl);
    pdat.resize(npart);

    

    {
        auto & pos_part = pdat.get_field<vec>(id_xyz).get_buf();
        sycl::host_accessor<vec> pos {*pos_part};

        auto & h_part = pdat.get_field<flt>(id_h).get_buf();
        sycl::host_accessor<flt> h {*h_part};

        for (u32 i = 0; i < npart; i ++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
            h[i] = 0.2 + distf(eng)*0.1;
        }
    }

    
    constexpr u32 reduc_level = 5;

    auto rtree = Radix_Tree<morton_mode, vec>(
            sycl_handler::get_compute_queue(), 
            {vec{-1,-1,-1},vec{1,1,1}},
            pdat.get_field<vec>(id_xyz).get_buf(), 
            npart , reduc_level
        );

    rtree.compute_cellvolume(sycl_handler::get_compute_queue());


    flt h_tol = 1.2;

    auto h_max = rtree.compute_field<flt>(
        sycl_handler::get_compute_queue(), 
        1,
        [&](sycl::handler &cgh,auto && node_looper){

            auto & h_part = pdat.get_field<flt>(id_h).get_buf();
            auto h = sycl::accessor{* h_part, cgh, sycl::read_only};

            node_looper(
                [=](auto && particle_looper, auto & buf, auto && get_id_store){
                    flt h_tmp = -1;

                    particle_looper([&](u32 particle_id){
                        f32 h_a = h[particle_id] * h_tol;
                        h_tmp   = (h_tmp > h_a ? h_tmp : h_a);
                    });

                    buf[get_id_store()] = h_tmp;
                }
            );

        },
        [&](auto && get_left_val, auto && get_right_val, auto & buf, auto && get_id_store){
            flt h_l = get_left_val(0);
            flt h_r = get_right_val(0);

            buf[get_id_store()] = (h_r > h_l ? h_r : h_l) ;
        }
    );


    


    auto compute_old = [&]() -> auto {
        logger::debug_sycl_ln("RadixTree", "compute int boxes");

        auto buf_cell_interact_rad = std::make_unique<sycl::buffer<flt>>(rtree.tree_internal_count + rtree.tree_leaf_count);
        sycl::range<1> range_leaf_cell{rtree.tree_leaf_count};

        sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
            u32 offset_leaf = rtree.tree_internal_count;

            auto h_max_cell = buf_cell_interact_rad->template get_access<sycl::access::mode::discard_write>(cgh);
            auto h          = pdat.get_field<flt>(id_h).get_buf()->template get_access<sycl::access::mode::read>(cgh);

            auto cell_particle_ids  = rtree.buf_reduc_index_map->template get_access<sycl::access::mode::read>(cgh);
            auto particle_index_map = rtree.buf_particle_index_map->template get_access<sycl::access::mode::read>(cgh);

            flt tol = h_tol;

            cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
                u32 gid = (u32)item.get_id(0);

                u32 min_ids = cell_particle_ids[gid];
                u32 max_ids = cell_particle_ids[gid + 1];
                f32 h_tmp   = 0;

                for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                    f32 h_a = h[particle_index_map[id_s]] * tol;
                    h_tmp   = (h_tmp > h_a ? h_tmp : h_a);
                }

                h_max_cell[offset_leaf + gid] = h_tmp;
            });
        });

        sycl::range<1> range_tree{rtree.tree_internal_count};
        auto ker_reduc_hmax = [&](sycl::handler &cgh) {
            u32 offset_leaf = rtree.tree_internal_count;

            auto h_max_cell = buf_cell_interact_rad->template get_access<sycl::access::mode::read_write>(cgh);

            auto rchild_id   = rtree.buf_rchild_id->get_access<sycl::access::mode::read>(cgh);
            auto lchild_id   = rtree.buf_lchild_id->get_access<sycl::access::mode::read>(cgh);
            auto rchild_flag = rtree.buf_rchild_flag->get_access<sycl::access::mode::read>(cgh);
            auto lchild_flag = rtree.buf_lchild_flag->get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                u32 gid = (u32)item.get_id(0);

                u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                flt h_l = h_max_cell[lid];
                flt h_r = h_max_cell[rid];

                h_max_cell[gid] = (h_r > h_l ? h_r : h_l);
            });
        };

        for (u32 i = 0; i < rtree.tree_depth; i++) {
            sycl_handler::get_compute_queue().submit(ker_reduc_hmax);
        }

        return std::move(buf_cell_interact_rad);
    };


    {
        auto old_vers = compute_old();

        u32 len = old_vers->size();

        sycl::host_accessor acc_old {*old_vers};
        sycl::host_accessor acc_new {*h_max.radix_tree_field_buf};

        for(u32 i = 0 ; i < len; i ++){
            Test_assert("same_buf", test_sycl_eq(acc_new[i],acc_old[i]));
        }
    }


}


constexpr auto list_npart_test = {1.14975700e+02,
       1.35304777e+02, 1.59228279e+02, 1.87381742e+02, 2.20513074e+02,
       2.59502421e+02, 3.05385551e+02, 3.59381366e+02, 4.22924287e+02,
       4.97702356e+02, 5.85702082e+02, 6.89261210e+02, 8.11130831e+02,
       9.54548457e+02, 1.12332403e+03, 1.32194115e+03, 1.55567614e+03,
       1.83073828e+03, 2.15443469e+03, 2.53536449e+03, 2.98364724e+03,
       3.51119173e+03, 4.13201240e+03, 4.86260158e+03, 5.72236766e+03,
       6.73415066e+03, 7.92482898e+03, 9.32603347e+03, 1.09749877e+04,
       1.29154967e+04, 1.51991108e+04, 1.78864953e+04, 2.10490414e+04,
       2.47707636e+04, 2.91505306e+04, 3.43046929e+04, 4.03701726e+04,
       4.75081016e+04, 5.59081018e+04, 6.57933225e+04, 7.74263683e+04,
       9.11162756e+04, 1.07226722e+05, 1.26185688e+05, 1.48496826e+05,
       1.74752840e+05, 2.05651231e+05, 2.42012826e+05, 2.84803587e+05,
       3.35160265e+05, 3.94420606e+05, 4.64158883e+05, 5.46227722e+05,
       6.42807312e+05, 7.56463328e+05, 8.90215085e+05, 1.04761575e+06,
       1.23284674e+06, 1.45082878e+06, 1.70735265e+06, 2.00923300e+06,
       2.36448941e+06, 2.78255940e+06, 3.27454916e+06, 3.85352859e+06,
       4.53487851e+06, 5.33669923e+06, 6.28029144e+06, 7.39072203e+06,
       8.69749003e+06, 1.02353102e+07};
       
       //, 1.20450354e+07, 1.41747416e+07,
       //1.66810054e+07, 1.96304065e+07, 2.31012970e+07, 2.71858824e+07,
       //3.19926714e+07, 3.76493581e+07, 4.43062146e+07, 5.21400829e+07,
       //6.13590727e+07, 7.22080902e+07, 8.49753436e+07, 1.00000000e+08};


Bench_start("tree field old compute performance", "treefieldcomputeperf_new", treefieldcomputeperf_new, 1){

    auto run_bench = [&](u32 npart, u32 reduc_level){

        using flt = f32;
        using vec = sycl::vec<flt,3>;
        using morton_mode = u32;
        using vec3i = morton_3d::morton_types<morton_mode>::int_vec_repr;

        std::mt19937 eng(0x1111);
        std::uniform_real_distribution<flt> distf(-1, 1);


        PatchDataLayout pdl;
        pdl.add_field<vec>("xyz", 1);
        pdl.add_field<flt>("h", 1);

        const auto id_xyz = pdl.get_field_idx<vec>("xyz");
        const auto id_h = pdl.get_field_idx<flt>("h");

        PatchData pdat(pdl);
        pdat.resize(npart);

        

        {
            auto & pos_part = pdat.get_field<vec>(id_xyz).get_buf();
            sycl::host_accessor<vec> pos {*pos_part};

            auto & h_part = pdat.get_field<flt>(id_h).get_buf();
            sycl::host_accessor<flt> h {*h_part};

            for (u32 i = 0; i < npart; i ++) {
                pos[i] = vec{distf(eng), distf(eng), distf(eng)};
                h[i] = 0.2 + distf(eng)*0.1;
            }
        }

        

        auto rtree = Radix_Tree<morton_mode, vec>(
                sycl_handler::get_compute_queue(), 
                {vec{-1,-1,-1},vec{1,1,1}},
                pdat.get_field<vec>(id_xyz).get_buf(), 
                npart , reduc_level
            );

        rtree.compute_cellvolume(sycl_handler::get_compute_queue());


        flt h_tol = 1.2;

        sycl_handler::get_compute_queue().wait();

        Timer timer; timer.start();

        auto h_max = rtree.compute_field<flt>(
            sycl_handler::get_compute_queue(), 
            1,
            [&](sycl::handler &cgh,auto && node_looper){

                auto & h_part = pdat.get_field<flt>(id_h).get_buf();
                auto h = sycl::accessor{* h_part, cgh, sycl::read_only};

                node_looper(
                    [=](auto && particle_looper, auto & buf, auto && get_id_store){
                        flt h_tmp = -1;

                        particle_looper([&](u32 particle_id){
                            f32 h_a = h[particle_id] * h_tol;
                            h_tmp   = (h_tmp > h_a ? h_tmp : h_a);
                        });

                        buf[get_id_store()] = h_tmp;
                    }
                );

            },
            [&](auto && get_left_val, auto && get_right_val, auto & buf, auto && get_id_store){
                flt h_l = get_left_val(0);
                flt h_r = get_right_val(0);

                buf[get_id_store()] = (h_r > h_l ? h_r : h_l) ;
            }
        );


        sycl_handler::get_compute_queue().wait();

        timer.end();

        Register_score("%result = " + std::to_string(npart) + "," + std::to_string(timer.nanosec));

    };


    for(u32 i = 0;i < 6; i++){

        Register_score("%tree_reduc = " + std::to_string(i));
        for(u32 cnt : list_npart_test){
            run_bench(cnt,i);
        }

    }
    

}




















Test_start("radix_tree", tree_comm, 2){



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


    constexpr u32 reduc_level = 5;

    auto rtree = Radix_Tree<morton_mode, vec>(
            sycl_handler::get_compute_queue(), 
            {vec{-1,-1,-1},vec{1,1,1}},
            pdat.get_field<vec>(id_xyz).get_buf(), 
            npart ,reduc_level
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


    auto cut = rtree.cut_tree(sycl_handler::get_compute_queue(), {vec{-1,-1,-1},vec{0,0.5,1}});


}









Test_start("radix_tree", treeleveljump_cell_range_test, 1){

    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;

    std::vector<vec> pos_part{
        vec{0,0,0},vec{0.05,0,0},
        vec{1,0,0},vec{0.95,0,0}
    };

    std::unique_ptr<sycl::buffer<vec>> xyz  = std::make_unique<sycl::buffer<vec>>(pos_part.data(),pos_part.size());


    auto rtree = Radix_Tree<morton_mode, vec>(
            sycl_handler::get_compute_queue(), 
            {vec{0,0,0},vec{1,1,1}},
            xyz, 
            4 ,0
        );

    rtree.compute_cellvolume(sycl_handler::get_compute_queue());

    {
        auto acc_min = sycl::host_accessor{*rtree.buf_pos_min_cell_flt};
        auto acc_max = sycl::host_accessor{*rtree.buf_pos_max_cell_flt};

        for(u32 i = 0 ; i < 7; i++){
            logger::raw_ln(i,acc_min[i],acc_max[i]);
        }
    }


}