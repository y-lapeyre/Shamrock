// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


/**
 * @file main.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "aliases.hpp"
#include "interfaces/interface_generator.hpp"
#include "interfaces/interface_handler.hpp"
#include "interfaces/interface_selector.hpp"
#include "io/dump.hpp"
#include "io/logs.hpp"
#include "particles/particle_patch_mover.hpp"
#include "patch/patch.hpp"
#include "patch/patch_field.hpp"
#include "patch/patch_reduc_tree.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patch/patchdata_exchanger.hpp"
#include "patch/patchdata_layout.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/loadbalancing_hilbert.hpp"
#include "patchscheduler/patch_content_exchanger.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sph/leapfrog.hpp"
#include "sph/sphpatch.hpp"
#include "sys/cmdopt.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "tree/radix_tree.hpp"
#include "unittests/shamrocktest.hpp"
#include "utils/string_utils.hpp"
#include "utils/time_utils.hpp"
#include <array>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include "sph/forces.hpp"
#include "sph/kernels.hpp"
#include "sph/sphpart.hpp"

class TestSimInfo {
  public:
    u32 stepcnt;
    f64 time;
};







template<class flt>
inline void correct_box_fcc(f32 r_particle, std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> & box){

    using vec3 = sycl::vec<flt, 3>;

    vec3 box_min = std::get<0>(box);
    vec3 box_max = std::get<1>(box);

    vec3 box_dim = box_max - box_min;

    vec3 iboc_dim = (box_dim / 
        vec3({
            2,
            sycl::sqrt(3.),
            2*sycl::sqrt(6.)/3
        }))/r_particle;

    u32 i = iboc_dim.x();
    u32 j = iboc_dim.y();
    u32 k = iboc_dim.z();

    //modify values to get even number on each axis to corect the periodicity
    if(i%2 == 1) i++;
    if(j%2 == 1) j++;
    if(k%2 == 1) k++;

    vec3 r_a = {
        2*i + 1,//((j+k) % 2), 
        sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
        2*sycl::sqrt(6.)*k/3
    };

    r_a.x() -=1;

    r_a *= r_particle;

    std::cout << "resizing box from (" <<
        box_dim.x() << ", " <<
        box_dim.y() << ", " <<
        box_dim.z() << ")" 
        << " to (" <<
        r_a.x() << ", " <<
        r_a.y() << ", " <<
        r_a.z() << ")" << std::endl;

    r_a += box_min;

    std::get<1>(box) =  r_a;

    
    
}

template<class flt,class Tpred_select,class Tpred_pusher>
inline void add_particles_fcc(
    flt r_particle, 
    std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> box,
    Tpred_select && selector,
    Tpred_pusher && part_pusher ){
    
    using vec3 = sycl::vec<flt, 3>;

    vec3 box_min = std::get<0>(box);
    vec3 box_max = std::get<1>(box);

    vec3 box_dim = box_max - box_min;

    vec3 iboc_dim = (box_dim / 
        vec3({
            2,
            sycl::sqrt(3.),
            2*sycl::sqrt(6.)/3
        }))/r_particle;

    std::cout << "len vector : (" << iboc_dim.x() << ", " << iboc_dim.y() << ", " << iboc_dim.z() << ")" << std::endl;

    for(u32 i = 0 ; i < iboc_dim.x(); i++){
        for(u32 j = 0 ; j < iboc_dim.y(); j++){
            for(u32 k = 0 ; k < iboc_dim.z(); k++){

                vec3 r_a = {
                    2*i + ((j+k) % 2),
                    sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
                    2*sycl::sqrt(6.)*k/3
                };

                r_a *= r_particle;
                r_a += box_min;

                if(selector(r_a)) part_pusher(r_a, r_particle);

            }
        }
    }

}







template<class flt, class Tgetter,class Tsetter,class Trho_x,class TM_x>
inline void strech_mapping_axis(
    flt x_min,
    flt x_max,
    
    u32 el_count,

    Tgetter getter,
    Tsetter setter,

    Trho_x rho_x,
    TM_x M_x){

    auto f_x = [&](flt x,flt x_0) -> flt{
        return (M_x(x)/ M_x(x_max)) - (x_0 - x_min)/(x_max-x_min);
    };

    auto fp_x = [&](flt x) -> flt{
        return rho_x(x)/M_x(x_max);
    };

    for(u32 i = 0; i < el_count; i++){

        flt x_0 = getter(i);
        flt x = x_0;
        while(!(sycl::fabs(f_x(x,x_0)) < 1e-6)){

            x -= f_x(x,x_0)/fp_x(x);

            //printf("-> %f %e\n",x,f_x(x,x_0));

        }

        setter(i,x);
        //printf("x_in : %f | x_out : %f | f : %e\n",x_0,x,sycl::fabs(f_x(x,x_0)));
    }

    

}









class TestTimestepper {
  public:

    

    static void init(SchedulerMPI &sched, TestSimInfo &siminfo) {

        f32_3 box_dim = {1,1,1};

        /*
        box_dim.x() *= 4;
        box_dim.y() /= 2;
        box_dim.z() /= 2;
        */

        std::tuple<f32_3,f32_3> box = {
            -box_dim,box_dim
        };

        

        f32 dr = 0.04;
        correct_box_fcc<f32>(dr,box);

        sched.set_box_volume<f32_3>(box);

        if (mpi_handler::world_rank == 0) {

            auto t = timings::start_timer("dumm setup", timings::timingtype::function);
            Patch p;

            
            p.node_owner_id = mpi_handler::world_rank;

            p.x_min = 0;
            p.y_min = 0;
            p.z_min = 0;

            p.x_max = HilbertLB::max_box_sz;
            p.y_max = HilbertLB::max_box_sz;
            p.z_max = HilbertLB::max_box_sz;

            p.pack_node_index = u64_max;

            PatchData pdat(sched.pdl);

            u32 ixyz = sched.pdl.get_field_idx<f32_3>("xyz");
            u32 ivxyz = sched.pdl.get_field_idx<f32_3>("vxyz");
            u32 iaxyz = sched.pdl.get_field_idx<f32_3>("axyz");
            u32 iaxyz_old = sched.pdl.get_field_idx<f32_3>("axyz_old");

            u32 ihpart = sched.pdl.get_field_idx<f32>("hpart");

            std::cout << "ixyz : " << ixyz << std::endl;
            std::cout << "ivxyz : " << ivxyz << std::endl;
            std::cout << "iaxyz : " << iaxyz << std::endl;
            std::cout << "iaxyz_old : " << iaxyz_old << std::endl;
            std::cout << "ihpart : " << ihpart << std::endl;

            

            add_particles_fcc(
                dr, 
                box , 
                [](f32_3 r){return true;}, 
                [&](f32_3 r,f32 h){
                    pdat.fields_f32_3[ixyz].insert_element(r); //r
                    //                      h    
                    pdat.fields_f32[ihpart].insert_element(h*2);
                    //                           v          a             a_old
                    pdat.fields_f32_3[ivxyz].insert_element(f32_3{0.f,0.f,0.f});
                    pdat.fields_f32_3[iaxyz].insert_element(f32_3{0.f,0.f,0.f});
                    pdat.fields_f32_3[iaxyz_old].insert_element(f32_3{0.f,0.f,0.f});
                });

            //std::cout << "tmp  " << pdat.fields_f32_3[ixyz].field_data.size();

            std::cout << "paticles count " << pdat.get_obj_cnt() << std::endl;

            //exit(0);

            if(false){
                /*
                f32 a = 0.01;

                f32 nmode = 1;
                strech_mapping_axis(std::get<0>(box).x(),std::get<1>(box).x(), pdat.get_obj_cnt(),

                    [&](u32 i) -> f32{
                        return pdat.fields_f32_3[ixyz].field_data[i].x();
                    }
                    ,
                    [&](u32 i,f32 r){
                        pdat.fields_f32_3[ixyz].field_data[i].x() = r;
                    },
                    
                    [&](f32 x) -> f32{

                        f32 x_min = std::get<0>(box).x();
                        f32 x_max = std::get<1>(box).x();
                        constexpr f32 pi = 3.141612;

                        return 1+a*sycl::cos(nmode*2.*pi*(x-x_min)/(x_max-x_min));
                    }, 
                    [&](f32 x) -> f32{

                        f32 xmin = std::get<0>(box).x();
                        f32 xmax = std::get<1>(box).x();

                        constexpr f32 pi = 3.141612;

                        return x - xmin + (a*(-xmax + xmin)* sycl::sin((nmode*2.*pi*(-x + xmin))/ (xmax - xmin)))/(nmode*2.*pi);
                    });

                */

                
            }

            p.data_count = pdat.get_obj_cnt();
            p.load_value = pdat.get_obj_cnt();

            /*
            p.data_count    = 1e6;
            p.load_value    = 1e6;
            std::mt19937 eng(0x1111);
            std::uniform_real_distribution<f32> distpos(-1, 1);

            for (u32 part_id = 0; part_id < p.data_count; part_id++){
                pdat.pos_s.emplace_back(f32_3{distpos(eng), distpos(eng), distpos(eng)}); //r
                //                      h    omega
                pdat.U1_s.emplace_back(0.02f);
                pdat.U1_s.emplace_back(0.00f);
                //                           v          a             a_old
                pdat.U3_s.emplace_back(f32_3{0.f,0.f,0.f});
                pdat.U3_s.emplace_back(f32_3{0.f,0.f,0.f});
                pdat.U3_s.emplace_back(f32_3{0.f,0.f,0.f});
            }
            */
                

            sched.add_patch(p, pdat);

            t.stop();

        } else {
            sched.patch_list._next_patch_id++;
        }
        mpi::barrier(MPI_COMM_WORLD);

        sched.owned_patch_id = sched.patch_list.build_local();

        // std::cout << sched.dump_status() << std::endl;
        sched.patch_list.build_global();
        // std::cout << sched.dump_status() << std::endl;

        //*
        sched.patch_tree.build_from_patchtable(sched.patch_list.global, HilbertLB::max_box_sz);
        //sched.patch_data.sim_box.min_box_sim_s = {-1};
        //sched.patch_data.sim_box.max_box_sim_s = {1};

        // std::cout << sched.dump_status() << std::endl;

        std::cout << "build local" << std::endl;
        sched.owned_patch_id = sched.patch_list.build_local();
        sched.patch_list.build_local_idx_map();
        sched.update_local_dtcnt_value();
        sched.update_local_load_value();

        // sched.patch_list.build_global();

        /*{
            SerialPatchTree<f32_3> sptree(sched.patch_tree, sched.get_box_tranform<f32_3>());
            sptree.attach_buf();

            PatchField<f32> h_field;
            h_field.local_nodes_value.resize(sched.patch_list.local.size());
            for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
                h_field.local_nodes_value[idx] = 0.02f;
            }
            h_field.build_global(mpi_type_f32);

            InterfaceHandler<f32_3, f32> interface_hndl;
            interface_hndl.compute_interface_list<InterfaceSelector_SPH<f32_3, f32>>(sched, sptree, h_field);
            interface_hndl.comm_interfaces(sched);
            interface_hndl.print_current_interf_map();

            // sched.dump_local_patches(format("patches_%d_node%d", 0, mpi_handler::world_rank));
        }*/
    }

    static void step(SchedulerMPI &sched, TestSimInfo &siminfo, std::string dump_folder) {

        SPHTimestepperLeapfrog<f32> leapfrog;

        SyCLHandler &hndl = SyCLHandler::get_instance();

        // std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);

        leapfrog.step(sched,dump_folder,siminfo.stepcnt,siminfo.time);
    }
};

template <class Timestepper, class SimInfo> class SimulationSPH {
  public:
    static void run_sim() {

        SyCLHandler &hndl = SyCLHandler::get_instance();

        PatchDataLayout pdl;

        pdl.xyz_mode = xyz32;
        pdl.add_field<f32_3>("xyz", 1);
        pdl.add_field<f32>("hpart", 1);
        pdl.add_field<f32_3>("vxyz",1);
        pdl.add_field<f32_3>("axyz",1);
        pdl.add_field<f32_3>("axyz_old",1);


        SchedulerMPI sched = SchedulerMPI(pdl,20000, 1);
        sched.init_mpi_required_types();

        logfiles::open_log_files();

        SimInfo siminfo;
        siminfo.time = 0;

        std::cout << " ------ init sim ------" << std::endl;

        auto t = timings::start_timer("init timestepper", timings::timingtype::function);
        Timestepper::init(sched, siminfo);
        t.stop();

        std::cout << " --- init sim done ----" << std::endl;



        std::filesystem::create_directory("step" + std::to_string(0));

        std::cout << "dumping state"<<std::endl;
        dump_state("step" + std::to_string(0) + "/", sched,siminfo.time);

        timings::dump_timings("### init_step ###");

        
        for (u32 stepi = 1; stepi < 500; stepi++) {

            if(stepi == 5){

                auto box = sched.get_box_volume<f32_3>();

                u32 ixyz = pdl.get_field_idx<f32_3>("xyz");
                u32 ivxyz = pdl.get_field_idx<f32_3>("vxyz");

                sched.for_each_patch_buf(
                    [&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto r = pdat_buf.fields_f32_3[ixyz]->get_access<sycl::access::mode::read>(cgh);
                            auto v = pdat_buf.fields_f32_3[ivxyz]->get_access<sycl::access::mode::discard_write>(cgh);

                            f32 deltv = 0.01;
                            u32 nmode = 2;
                            constexpr f32 pi = 3.141612;
                            f32 z_min = std::get<0>(box).z();
                            f32 z_max = std::get<1>(box).z();

                            cgh.parallel_for( sycl::range{pdat_buf.element_count}, [=](sycl::item<1> item) { 

                                f32 z = r[item].z();

                                v[item] = {
                                        0,
                                        0,
                                        deltv*sycl::cos(nmode*2.*pi*(z-z_min)/(z_max-z_min))
                                    }
                                ; 
                            });
                        });

                    }
                );
            }


            std::cout << " ------ step time = " << stepi << " ------" << std::endl;

            std::cout << "time : " << siminfo.time << std::endl;

            std::filesystem::create_directory("step" + std::to_string(stepi));

            siminfo.stepcnt = stepi;

            auto step_timer = timings::start_timer("timestepper step", timings::timingtype::function);
            
            Timestepper::step(sched, siminfo,"step" + std::to_string(stepi));

            step_timer.stop();

            

            dump_state("step" + std::to_string(stepi) + "/", sched,siminfo.time);

            timings::dump_timings("### "
                                  "step" +
                                  std::to_string(stepi) + " ###");
        }

        logfiles::close_log_files();

        sched.free_mpi_required_types();
    }
};



int main(int argc, char *argv[]) {



    auto test_l = [](int a){
        return a;
    };


    static_assert(std::is_same<decltype(test_l(0)), int>::value, "retval must be bool");




    std::cout << shamrock_title_bar_big << std::endl;

    mpi_handler::init();

    Cmdopt &opt = Cmdopt::get_instance();
    opt.init(argc, argv, "./shamrock");

    SyCLHandler &hndl = SyCLHandler::get_instance();
    hndl.init_sycl();







    SimulationSPH<TestTimestepper, TestSimInfo>::run_sim();









    mpi_handler::close();
}