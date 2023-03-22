// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/memory/memory.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/legacy/utils/time_utils.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include "shamalgs/random/random.hpp"

#include "shamrock/tree/tests/TreeTests.hpp"
#include "shamrock/sfc/MortonKernels.hpp"
#include "shamrock/sfc/morton.hpp"



auto get_Nmax = []() -> f64 {
    return  1e8 * 1;
};




template<class morton_mode, class flt, u32 reduc_lev>
inline void test_tree_build_steps(std::string dset_name) {

    using vec = sycl::vec<flt, 3>;

    f64 Nmax_flt = get_Nmax();

    u32 Nmax = u32(sycl::fmin(Nmax_flt, 2e9));

    auto coord_range = get_test_coord_ranges<vec>();

    auto pos =
        shamalgs::random::mock_buffer_ptr<vec>(0x111, Nmax, coord_range.lower, coord_range.upper);


    shamalgs::memory::move_buffer_on_queue(
        shamsys::instance::get_compute_queue(), 
        *pos);

    std::vector<f64> times_morton              ;
    std::vector<f64> times_reduc               ;
    std::vector<f64> times_karras              ;
    std::vector<f64> times_compute_int_range   ;
    std::vector<f64> times_compute_coord_range ;
    std::vector<f64> times_morton_build        ;
    std::vector<f64> times_trailling_fill      ;
    std::vector<f64> times_index_gen           ;
    std::vector<f64> times_morton_sort         ;
    std::vector<f64> times_full_tree           ;

    std::vector<f64> Npart;

    for (f64 cnt = 1000; cnt < Nmax; cnt *= 1.1) {
        Npart.push_back(u32(cnt));
    }

    for (f64 cnt : Npart) {
        times_morton             .push_back(0);
        times_reduc              .push_back(0);
        times_karras             .push_back(0);
        times_compute_int_range  .push_back(0);
        times_compute_coord_range.push_back(0);
        times_morton_build       .push_back(0);
        times_trailling_fill     .push_back(0);
        times_index_gen          .push_back(0);
        times_morton_sort        .push_back(0);
        times_full_tree          .push_back(0);
    }

    auto get_repetition_count = [](f64 cnt){
        if(cnt < 1e5) return 100;
        return 20;
    };

    u32 index = 0;
    for (f64 cnt : Npart) {
        logger::debug_ln("TestTreePerf", cnt, dset_name);
        for (u32 rep_count = 0; rep_count < get_repetition_count(cnt); rep_count++) {
        
            
            
            Timer timer;
            u32 cnt_obj = cnt;

            auto time_func = [](auto f){
                shamsys::instance::get_compute_queue().wait();
                Timer timer;
                timer.start();

                f();
                shamsys::instance::get_compute_queue().wait();

                timer.end();
                return timer.nanosec / 1.e9;
            };

            {
                shamrock::tree::TreeMortonCodes<morton_mode> tree_morton_codes;
                shamrock::tree::TreeReducedMortonCodes<morton_mode> tree_reduced_morton_codes;
                shamrock::tree::TreeStructure<morton_mode> tree_struct;

                
                times_morton[index]+=(
                    time_func([&](){
                        tree_morton_codes.build(shamsys::instance::get_compute_queue(), coord_range, cnt_obj, *pos);
                    })
                );


                bool one_cell_mode;
                times_reduc[index]+=(time_func([&](){
            
                    tree_reduced_morton_codes.build(
                        shamsys::instance::get_compute_queue(),cnt_obj,reduc_lev,tree_morton_codes,one_cell_mode
                    );

                }));


                times_karras[index]+=(time_func([&](){

                    if (!one_cell_mode) {
                        tree_struct.build(shamsys::instance::get_compute_queue(), tree_reduced_morton_codes.tree_leaf_count - 1, *tree_reduced_morton_codes.buf_tree_morton);
                    } else {
                        tree_struct.build_one_cell_mode();
                    }

                }));
            }


            {
                RadixTree<morton_mode, vec, 3> rtree = RadixTree<morton_mode, vec, 3>(
                    shamsys::instance::get_compute_queue(),
                    {coord_range.lower, coord_range.upper},
                    pos,
                    cnt,
                    reduc_lev
                );

                times_compute_int_range[index]+=(time_func([&](){

                    rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
                }));

                times_compute_coord_range[index]+=(time_func([&](){

                
                    rtree.convert_bounding_box(shamsys::instance::get_compute_queue());
                    

                }));
            }

            {

                using namespace shamrock::sfc;

                u32 morton_len = shambase::roundup_pow2_clz(cnt_obj);


                auto out_buf_morton = std::make_unique<sycl::buffer<morton_mode>>(morton_len);


                times_morton_build[index]+=(time_func([&](){

                    MortonKernels<morton_mode, vec, 3>::sycl_xyz_to_morton(
                        shamsys::instance::get_compute_queue(),
                        cnt_obj,
                        *pos,
                        coord_range.lower,
                        coord_range.upper,
                        out_buf_morton
                    );
                }));



                times_trailling_fill[index]+=(time_func([&](){
                
                    MortonKernels<morton_mode, vec, 3>::sycl_fill_trailling_buffer(shamsys::instance::get_compute_queue(), cnt_obj, morton_len, out_buf_morton);
                
                }));

                std::unique_ptr<sycl::buffer<u32>> out_buf_particle_index_map;
                
                times_index_gen[index]+=(time_func([&](){
                
                    out_buf_particle_index_map = std::make_unique<sycl::buffer<u32>>(
                        shamalgs::algorithm::gen_buffer_index(shamsys::instance::get_compute_queue(), morton_len)
                    );
                }));

                times_morton_sort[index]+=(time_func([&](){
                    sycl_sort_morton_key_pair(shamsys::instance::get_compute_queue(), morton_len, out_buf_particle_index_map, out_buf_morton);
                }));


            }

            {
                shamsys::instance::get_compute_queue().wait();
                Timer timer2;
                timer2.start();
                
                RadixTree<morton_mode, vec, 3> rtree = RadixTree<morton_mode, vec, 3>(
                    shamsys::instance::get_compute_queue(),
                    {coord_range.lower, coord_range.upper},
                    pos,
                    cnt,
                    reduc_lev
                );

                rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
                rtree.convert_bounding_box(shamsys::instance::get_compute_queue());
                shamsys::instance::get_compute_queue().wait();
                timer2.end();
                times_full_tree[index]+=( timer2.nanosec / 1.e9);
            }

        }

        index ++;
    }

    


    index = 0;
    for (f64 cnt : Npart) {

        times_morton             [index]/= get_repetition_count(cnt);
        times_reduc              [index]/= get_repetition_count(cnt);
        times_karras             [index]/= get_repetition_count(cnt);
        times_compute_int_range  [index]/= get_repetition_count(cnt);
        times_compute_coord_range[index]/= get_repetition_count(cnt);
        times_morton_build       [index]/= get_repetition_count(cnt);
        times_trailling_fill     [index]/= get_repetition_count(cnt);
        times_index_gen          [index]/= get_repetition_count(cnt);
        times_morton_sort        [index]/= get_repetition_count(cnt);
        times_full_tree          [index]/= get_repetition_count(cnt);

        index++;
    }



    auto &dset = shamtest::test_data().new_dataset(dset_name);

    dset.add_data("Npart", Npart);

    dset.add_data("times_morton",              times_morton                   );
    dset.add_data("times_reduc",               times_reduc                    );
    dset.add_data("times_karras",              times_karras                   );
    dset.add_data("times_compute_int_range",   times_compute_int_range        );
    dset.add_data("times_compute_coord_range", times_compute_coord_range      );
    dset.add_data("times_morton_build",        times_morton_build             );
    dset.add_data("times_trailling_fill",      times_trailling_fill           );
    dset.add_data("times_index_gen",           times_index_gen                );
    dset.add_data("times_morton_sort",         times_morton_sort              );
    dset.add_data("times_full_tree",           times_full_tree              );
}













TestStart(Benchmark, "shamrock_article1:tree_build_perf", tree_building_paper_results_tree_perf_steps, 1){

    test_tree_build_steps<u32, f32, 0>("morton = u32, field type = f32");
    test_tree_build_steps<u64, f32, 0>("morton = u64, field type = f32");
    test_tree_build_steps<u32, f64, 0>("morton = u32, field type = f64");
    test_tree_build_steps<u64, f64, 0>("morton = u64, field type = f64");
    test_tree_build_steps<u32, u64, 0>("morton = u32, field type = u64");
    test_tree_build_steps<u64, u64, 0>("morton = u64, field type = u64");

}