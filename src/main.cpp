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
#include "core/io/dump.hpp"
#include "core/io/logs.hpp"
#include "core/patch/base/patch.hpp"
#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "core/patch/comm/patch_content_exchanger.hpp"
#include "core/patch/comm/patch_object_mover.hpp"
#include "core/patch/comm/patchdata_exchanger.hpp"
#include "core/patch/interfaces/interface_generator.hpp"
#include "core/patch/interfaces/interface_handler.hpp"
#include "core/patch/interfaces/interface_selector.hpp"
#include "core/patch/patchdata_buffer.hpp"
#include "core/patch/scheduler/loadbalancing_hilbert.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/patch/utility/patch_field.hpp"
#include "core/patch/utility/patch_reduc_tree.hpp"
#include "core/patch/utility/serialpatchtree.hpp"
#include "core/sys/cmdopt.hpp"
#include "core/sys/mpi_handler.hpp"
#include "core/sys/sycl_mpi_interop.hpp"
#include "core/tree/radix_tree.hpp"
#include "core/utils/string_utils.hpp"
#include "core/utils/time_utils.hpp"
#include "models/generic/physics/units.hpp"
#include "models/generic/setup/SPHSetup.hpp"
#include "models/sph/base/kernels.hpp"
#include "models/sph/base/kernels.hpp"
#include "models/sph/base/sphpart.hpp"
#include "models/sph/forces.hpp"
#include "models/sph/gas_sync.hpp"
#include "models/sph/leapfrog.hpp"
#include "models/sph/models/gas_only.hpp"
#include "models/sph/models/gas_only_intu.hpp"
#include "models/sph/models/gas_only_visco.hpp"
#include "models/sph/sphpatch.hpp"
#include "runscript/rscripthandler.hpp"
#include "unittests/shamrocktest.hpp"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

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







f32 part_mass;

class TestTimestepper {
  public:

    

    static void init(PatchScheduler &sched, TestSimInfo &siminfo) {

        

        
        /*soundwave

        f32_3 box_dim = {1,1,1};

        box_dim.y() /= 4;

        std::tuple<f32_3,f32_3> box = {
            -box_dim,box_dim
        };
        
        f32 dr = 0.04; //for soundwave
        
        correct_box_fcc<f32>(dr,box);

        sched.set_box_volume<f32_3>(box);

        SPHSetup<f32> setup(sched,true);

        setup.init_setup();


        setup.add_particules_fcc(dr, box, [](f32_3 r){return true;});
        */

        //sedov
        f32_3 box_dim = {1,1,1};

        box_dim.y() /= 32;
        box_dim.x() /= 32;

        std::tuple<f32_3,f32_3> box = {
            -box_dim,box_dim
        };
        
        f32 dr = 0.0025; //for soundwave
        
        correct_box_fcc<f32>(dr,box);

        sched.set_box_volume<f32_3>(box);

        SPHSetup<f32> setup(sched,true);

        setup.init_setup();


        setup.add_particules_fcc(dr, box, [&](f32_3 r){return sycl::fabs(r.z()) < box_dim.z()/1.99;});
        setup.add_particules_fcc(dr*2, box, [&](f32_3 r){return sycl::fabs(r.z()) >= box_dim.z()/1.99;});

        part_mass = setup.get_part_mass(0.0010986328125);
        //part_mass = 7.783467665607285e-08;

        for (auto & [pid,pdat] : sched.patch_data.owned_data) {

            PatchDataField<f32_3> & xyz = pdat.get_field<f32_3>(sched.pdl.get_field_idx<f32_3>("xyz"));

            PatchDataField<f32> & f = pdat.template get_field<f32>(sched.pdl.get_field_idx<f32>("u"));

            for (u32 i =0; i <f.size() ; i++) {
                f32_3 r = xyz.usm_data()[i] ;

                if (sycl::fabs(r.z()) < box_dim.z()/2){
                    f.usm_data()[i] = 1/(((5./3.) - 1)*1);
                }else{
                    f.usm_data()[i] = 0.1/(((5./3.) - 1)*0.125);
                }
            }

            //f.override(dr);
        }

    }

    static void step(PatchScheduler &sched, TestSimInfo &siminfo, std::string dump_folder) {

        using namespace models::sph;

        //GasOnlyLeapfrog<f32, u32, kernels::M4<f32>> leapfrog;
        //GasOnlyViscoLeapfrog<f32, u32, kernels::M4<f32>> leapfrog;
        GasOnlyInternalU<f32, u32, kernels::M4<f32>> leapfrog;


        const f32 htol_up_tol  = 1.4;
        const f32 htol_up_iter = 1.2;

        const f32 cfl_cour  = 0.02;
        const f32 cfl_force = 0.3;

        //const flt eos_cs = 1;

        bool do_force = siminfo.stepcnt > 4;
        bool do_corrector = siminfo.stepcnt > 5;

        leapfrog.htol_up_tol = htol_up_tol;
        leapfrog.htol_up_iter = htol_up_iter;
        leapfrog.cfl_cour = cfl_cour;
        leapfrog.cfl_force = cfl_force;
        leapfrog.do_force = do_force;
        leapfrog.do_corrector = do_corrector;

        leapfrog.gpart_mass = part_mass;


        //SPHTimestepperLeapfrogIsotGas<f32> leapfrog;

        SyCLHandler &hndl = SyCLHandler::get_instance();

        // std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);

        leapfrog.step(sched,siminfo.time);

        //leapfrog.step(sched,dump_folder,siminfo.stepcnt,siminfo.time);
    }
};



class TestTimestepperSync {
  public:

    SPHTimestepperLeapfrogIsotGasSync<f32> leapfrog;

    void init(PatchScheduler &sched, TestSimInfo &siminfo) {


        f32_3 box_dim = {1,1,1};

        std::tuple<f32_3,f32_3> box = {
            -box_dim,box_dim
        };

        

        f32 dr = 0.04;
        correct_box_fcc<f32>(dr,box);

        sched.set_box_volume<f32_3>(box);

        SPHSetup<f32> setup(sched,true);

        setup.init_setup();
        setup.add_particules_fcc(dr, box, [](f32_3 r){return true;});

        leapfrog.sync_parts.push_back(
            SPHTimestepperLeapfrogIsotGasSync<f32>::SyncPart{
                {0,0,0},
                {0,0,0},
                1.e11
            });
    }

    void step(PatchScheduler &sched, TestSimInfo &siminfo, std::string dump_folder) {

        

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

        // disable u for soundwave
        pdl.add_field<f32>("u", 1);
        pdl.add_field<f32>("du",1);
        pdl.add_field<f32>("du_old",1);



        //PatchScheduler sched = PatchScheduler(pdl,20000, 1); //soundwave test
        PatchScheduler sched = PatchScheduler(pdl,100000, 1);
        sched.init_mpi_required_types();

        logfiles::open_log_files();

        SimInfo siminfo;
        siminfo.time = 0;

        std::cout << " ------ init sim ------" << std::endl;

        auto t = timings::start_timer("init timestepper", timings::timingtype::function);
        Timestepper stepper;
        stepper.init(sched, siminfo);
        t.stop();

        std::cout << " --- init sim done ----" << std::endl;



        std::filesystem::create_directory("step" + std::to_string(0));

        std::cout << "dumping state"<<std::endl;
        dump_state("step" + std::to_string(0) + "/", sched,siminfo.time);

        timings::dump_timings("### init_step ###");

        
        for (u32 stepi = 1; stepi < 1e4; stepi++) {


            /* //wave setup
            if(stepi == 5 && true){

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
            */

            if(stepi == 5 && false){

                auto box = sched.get_box_volume<f32_3>();

                u32 ixyz = pdl.get_field_idx<f32_3>("xyz");
                u32 ivxyz = pdl.get_field_idx<f32_3>("vxyz");

                sched.for_each_patch_buf(
                    [&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto r = pdat_buf.fields_f32_3[ixyz]->get_access<sycl::access::mode::read>(cgh);
                            auto v = pdat_buf.fields_f32_3[ivxyz]->get_access<sycl::access::mode::discard_write>(cgh);

                            f32 deltv = 1e-4;
                            u32 nmode = 4;
                            constexpr f32 pi = 3.141612;
                            f32 x_min = std::get<0>(box).x();
                            f32 x_max = std::get<1>(box).x();


                            cgh.parallel_for( sycl::range{pdat_buf.element_count}, [=](sycl::item<1> item) { 
                                u32 i = item.get_id(0);
                                f32 x = r[item].x();
                                f32 z = r[item].z();

                                f32 pert = 0;


                                if(z > 0){
                                    pert = 1; 
                                }else{
                                    pert = -1;
                                }pert *= 0.1;


                                v[item] = {
                                        pert,
                                        0,
                                        deltv*sycl::cos(nmode*2.*pi*(x-x_min)/(x_max-x_min))
                                    }
                                ;

                                
                            });
                        });

                    }
                );
            }


            std::cout << " ------ step time = " << stepi << " ------" << std::endl;

            std::cout << "time : " << siminfo.time << std::endl;


            //std::filesystem::create_directory("step" + std::to_string(stepi));
            
            u32 skip_cnt = 20;

            //*
            if(stepi % skip_cnt == 0){
                std::filesystem::create_directory("step" + std::to_string(stepi/skip_cnt));
            }
            //*/

            siminfo.stepcnt = stepi;

            auto step_timer = timings::start_timer("timestepper step", timings::timingtype::function);
            
            stepper.step(sched, siminfo,"step" + std::to_string(stepi));

            step_timer.stop();

            //dump_state("step" + std::to_string(stepi) + "/", sched,siminfo.time);
            //*
            if(stepi % skip_cnt == 0){
                dump_state("step" + std::to_string(stepi/skip_cnt) + "/", sched,siminfo.time);
            }
            //*/
            

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

    //*
    {
        RunScriptHandler rscript;
        rscript.run_ipython();
    }
    //*/





    using namespace units;

    Units<f64> code_units(
        yr_s,
        au_m,
        earth_mass_kg,
        1,
        1,
        1,
        1
    );

    //to init values in code
    f64 planet_mass = 2*code_units.jupiter_mass;

    std::cout << "planet mass : " << planet_mass << " code_unit mass" << std::endl;

    std::cout << "planet mass : " << planet_mass/code_units.jupiter_mass << " " << get_symbol(jupiter_mass) << std::endl;





    SimulationSPH<TestTimestepper, TestSimInfo>::run_sim();
    //SimulationSPH<TestTimestepperSync, TestSimInfo>::run_sim();








    mpi_handler::close();
}