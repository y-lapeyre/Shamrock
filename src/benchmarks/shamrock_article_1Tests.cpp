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
#include "shamrock/tree/TreeStructureWalker.hpp"
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






template<class u_morton, class flt>
class SPHTestInteractionCrit {
    using vec = sycl::vec<flt, 3>;
    public:

    RadixTree<u_morton, vec, 3> &tree;
    sycl::buffer<vec> &positions;
    u32 part_count;
    flt Rpart;

    class Access {
        public:
        sycl::accessor<vec, 1, sycl::access::mode::read> part_pos;
        flt Rpart;
        flt Rpart_pow2;

        Access(SPHTestInteractionCrit crit, sycl::handler &cgh)
            : part_pos{crit.positions, cgh, sycl::read_only}, Rpart(crit.Rpart),Rpart_pow2(crit.Rpart*crit.Rpart) {}

        class Values {
            public:
            vec xyz_a;
            Values(Access acc, u32 index)
                : xyz_a(acc.part_pos[index]) {}
        };
    };

    class TreeFieldAccess {
        public:
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_min;
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_max;

        TreeFieldAccess(SPHTestInteractionCrit crit, sycl::handler &cgh)
            : tree_cell_coordrange_min{*crit.tree.buf_pos_min_cell_flt, cgh, sycl::read_only},
                tree_cell_coordrange_max{
                    *crit.tree.buf_pos_max_cell_flt, cgh, sycl::read_only} {}
    };

    static bool
    criterion(u32 node_index, TreeFieldAccess tree_acc, typename Access::Values current_values, Access int_accessors) {
        vec cur_pos_min_cell_b = tree_acc.tree_cell_coordrange_min[node_index];
        vec cur_pos_max_cell_b = tree_acc.tree_cell_coordrange_max[node_index];

        vec box_int_sz = {int_accessors.Rpart,int_accessors.Rpart,int_accessors.Rpart};

        return 
            BBAA::cella_neigh_b(
                current_values.xyz_a - box_int_sz, current_values.xyz_a + box_int_sz, 
                cur_pos_min_cell_b, cur_pos_max_cell_b) ||
            BBAA::cella_neigh_b(
                current_values.xyz_a, current_values.xyz_a,                   
                cur_pos_min_cell_b - box_int_sz, cur_pos_min_cell_b + box_int_sz);
    };
};




template<class morton_mode, class flt, u32 reduc_lev>
void test_sph_iter_overhead(std::string dset_name){

    sycl::queue & q = shamsys::instance::get_compute_queue();

    //setup the particle distribution

    using vec = sycl::vec<flt, 3>;

    f64 Nmax_flt = get_Nmax();

    u32 Nmax = 2U<<27U;

    auto coord_range = get_test_coord_ranges<vec>();

    std::vector<f64> Npart;
    std::vector<f64> avg_neigh;
    std::vector<f64> var_neigh;
    std::vector<f64> rpart_vec;
    std::vector<f64> times;


    auto mix_seed = [](f64 seed) -> u32 {
        f64 a = 16807;
        f64 m = 2147483647;
        seed = std::fmod((a * seed) , m);
        return u32_max*(seed / m);
    };

    u32 test_per_n = 10;
    u32 seed = 0x111;
    for (f64 cnt = 1000; cnt < Nmax; cnt *= 1.1) {
        for(u32 i = 0; i < 15; i++){
            seed = mix_seed(seed);
            u32 len_pos = cnt;



            flt volume_per_obj = coord_range.get_volume()/len_pos;

            flt len_per_obj = sycl::cbrt(volume_per_obj);


            flt rpart = 0;
            {
                std::mt19937 eng (seed);
                rpart = std::uniform_real_distribution<flt>(0,len_per_obj*4)(eng);
            }

            

            logger::debug_ln("TestTreePerf", 
                shambase::format("dataset : {}, len={:e} seed={:10} len_p_obj={:e} rpart={:e}",
                                    dset_name,f32(len_pos)  ,seed     ,len_per_obj   ,rpart)
                );

            
            


            auto pos =
            shamalgs::random::mock_buffer_ptr<vec>(seed, len_pos, coord_range.lower, coord_range.upper);

            


            sycl::buffer<u32> neighbours(len_pos);

            //try{
                RadixTree<morton_mode, vec, 3> rtree = RadixTree<morton_mode, vec, 3>(
                    shamsys::instance::get_compute_queue(),
                    {coord_range.lower, coord_range.upper},
                    pos,
                    cnt,
                    reduc_lev
                );

                rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
                rtree.convert_bounding_box(shamsys::instance::get_compute_queue());

                auto benchmark = [&]() -> f64 {
                    Timer t;
                    
                    q.wait();
                    t.start();

                    using Criterion = SPHTestInteractionCrit<morton_mode,flt>;
                    using CriterionAcc = typename Criterion::Access;
                    using CriterionVal = typename CriterionAcc::Values; 

                    using namespace shamrock::tree;
                    TreeStructureWalker walk = generate_walk<Recompute>(
                        rtree.tree_struct, len_pos, SPHTestInteractionCrit<morton_mode,flt>{rtree,*pos, len_pos, rpart}
                    );


                    q.submit([&](sycl::handler &cgh) {
                        auto walker        = walk.get_access(cgh);
                        auto leaf_iterator = rtree.get_leaf_access(cgh);

                        sycl::accessor neigh_count {neighbours,cgh,sycl::write_only, sycl::no_init};
                        

                        cgh.parallel_for(walker.get_sycl_range(), [=](sycl::item<1> item) {
                            u32 sum = 0;

                            CriterionVal int_values{walker.criterion(), static_cast<u32>(item.get_linear_id())};

                            walker.for_each_node(
                                item,int_values,
                                [&](u32 /*node_id*/, u32 leaf_iterator_id) {
                                    leaf_iterator.iter_object_in_leaf(
                                        leaf_iterator_id, [&](u32 obj_id) { 

                                            vec xyz_b = walker.criterion().part_pos[obj_id];
                                            vec dxyz = xyz_b - int_values.xyz_a;
                                            flt dot_ = sycl::dot(dxyz,dxyz);

                                            if(dot_ < walker.criterion().Rpart_pow2){
                                                sum += 1; 
                                            }
                                        }
                                    );
                                },
                                [&](u32 node_id) {}
                            );

                            neigh_count[item] = sum;
                        });
                        
                    });

                    q.wait();
                    t.end();

                    return t.nanosec * 1e-9;
                };

                f64 time = benchmark();

                {
                    
                    f64 npart = len_pos;
                    f64 neigh_avg = 0;
                    f64 neigh_var = 0;

                    {
                        sycl::host_accessor acc{neighbours, sycl::read_only};
                        for(u32 i = 0; i < len_pos; i++){
                            neigh_avg += acc[i];
                        }
                        neigh_avg /= len_pos;
                        
                        for(u32 i = 0; i < len_pos; i++){
                            neigh_var += (acc[i]- neigh_avg)*(acc[i]- neigh_avg);
                        }
                        neigh_var /= len_pos;
                    }

                    Npart.push_back(npart);
                    avg_neigh.push_back(neigh_avg);
                    var_neigh.push_back(neigh_var);
                    times.push_back(time);
                    rpart_vec.push_back(rpart);

                }
            //}catch(...){
            //    continue;
            //}
        }

    }


    auto & dat_test = shamtest::test_data().new_dataset(dset_name);

    dat_test.add_data("Nobj", Npart);
    dat_test.add_data("avg_neigh", avg_neigh);
    dat_test.add_data("var_neigh", var_neigh);
    dat_test.add_data("time", times);
    dat_test.add_data("rpart", rpart_vec);

}

TestStart(Benchmark, "shamrock_article1:sph_walk_perf", tree_walk_sph_paper_results_tree_perf_steps, 1){
    test_sph_iter_overhead<u32, f32, 15>("uniform distrib reduction level 15");
    test_sph_iter_overhead<u32, f32, 6>("uniform distrib reduction level 6");
    test_sph_iter_overhead<u32, f32, 3>("uniform distrib reduction level 3");
    test_sph_iter_overhead<u32, f32, 0>("uniform distrib no reduction");
}