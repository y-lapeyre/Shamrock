#include "aliases.hpp"

#include "core/patch/base/patchdata_layout.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "models/sph/models/basic_sph_gas.hpp"
#include "models/sph/setup/sph_setup.hpp"
#include "unittests/shamrocktest.hpp"

template<class flt> 
std::tuple<f64,f64> benchmark_periodic_box(f32 dr, u32 npatch){

    using vec = sycl::vec<flt,3>;

    u64 Nesti = (2.F/dr)*(2.F/dr)*(2.F/dr);

    PatchDataLayout pdl;

    pdl.xyz_mode = xyz32;
    pdl.add_field<f32_3>("xyz", 1);
    pdl.add_field<f32>("hpart", 1);
    pdl.add_field<f32_3>("vxyz",1);
    pdl.add_field<f32_3>("axyz",1);
    pdl.add_field<f32_3>("axyz_old",1);

    PatchScheduler sched = PatchScheduler(pdl,Nesti/npatch, 1);
    sched.init_mpi_required_types();


    using Setup = models::sph::SetupSPH<f32, models::sph::kernels::M4<f32>>;

    Setup setup;
    setup.init(sched);

    auto box = setup.get_ideal_box(dr, {vec{-1,-1,-1},vec{1,1,1}});
    std::get<0>(box).x() -= 1e-6;
    std::get<0>(box).y() -= 1e-6;
    std::get<0>(box).z() -= 1e-6;
    std::get<1>(box).x() += 1e-6;
    std::get<1>(box).y() += 1e-6;
    std::get<1>(box).z() += 1e-6;
    sched.set_box_volume(box);

    setup.set_boundaries(true);
    setup.add_particules_fcc(sched, dr, box);
    setup.set_total_mass(8.);

    auto pmass = setup.get_part_mass();

    auto Npart = 8./pmass;

    for(u32 i = 0; i < 5; i++){
        setup.update_smoothing_lenght(sched);
    }


    if(sched.patch_list.global.size() != npatch){
        throw "pute";
    }

    

    using Model = models::sph::BasicSPHGas<f32, models::sph::kernels::M4<f32>>;

    Model model ;

    const f32 htol_up_tol  = 1.4;
    const f32 htol_up_iter = 1.2;

    const f32 cfl_cour  = 0.02;
    const f32 cfl_force = 0.3;

    model.set_cfl_force(0.3);
    model.set_cfl_cour(0.02);
    model.set_particle_mass(pmass);

    sycl_handler::get_compute_queue().wait();

    Timer t;
    t.start();
    
    model.evolve(sched, 0, 1e-3);
    sycl_handler::get_compute_queue().wait();

    t.end();

    return {Npart,t.nanosec/1e9};

}

template<class flt>
void benchmark_periodic_box_main(u32 npatch, std::string name){

    auto & dset = shamrock::test::test_data().new_dataset(name);

    std::vector<f64> npart;
    std::vector<f64> times;

    f32 dr = 0.05;
    for(; dr > 0.004; dr /= 1.1){
        auto [N,t] = benchmark_periodic_box<flt>(dr, npatch);
        npart.push_back(N);
        times.push_back(t);
    }

    dset.add_data("Npart", npart);
    dset.add_data("times", times);

}

TestStart(Benchmark, "benchmark periodic box sph", bench_per_box_sph, -1){

    benchmark_periodic_box_main<f32>(1,"partch_1");
    benchmark_periodic_box_main<f32>(8,"partch_8");
    benchmark_periodic_box_main<f32>(64,"partch_64");

}