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
#include "patch/patchdata_exchanger.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/loadbalancing_hilbert.hpp"
#include "patchscheduler/patch_content_exchanger.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sys/cmdopt.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "tree/radix_tree.hpp"
#include "unittests/shamrocktest.hpp"
#include "utils/string_utils.hpp"
#include "utils/time_utils.hpp"
#include <array>
#include <memory>
#include <mpi.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "sph/kernels.hpp"
#include "sph/sphpart.hpp"
#include "sph/forces.hpp"

class TestSimInfo {
  public:
    u32 time;
};


class CurDataLayout{public:

    using pos_type = f32;
    class U1_s{public:
        static constexpr u32 nvar = 2;

        static constexpr u32 ihpart = 0;
        static constexpr u32 iomega = 1;
    };

    class U1_d{public:
        static constexpr u32 nvar = 0;
    };

    class U3_s{public:
        static constexpr u32 nvar = 3;

        static constexpr u32 ivxyz = 0;
        static constexpr u32 iaxyz = 1;
        static constexpr u32 iaxyz_old = 2;
    };

    class U3_d{public:
        static constexpr u32 nvar = 0;
    };
};


class TestTimestepper {
  public:
    static void init(SchedulerMPI &sched, TestSimInfo &siminfo) {

        patchdata_layout::set(1, 0, 2, 0, 3, 0);
        patchdata_layout::sync(MPI_COMM_WORLD);

        if (mpi_handler::world_rank == 0) {

            auto t = timings::start_timer("dumm setup", timings::timingtype::function);
            Patch p;

            p.data_count    = 1e7;
            p.load_value    = 1e7;
            p.node_owner_id = mpi_handler::world_rank;

            p.x_min = 0;
            p.y_min = 0;
            p.z_min = 0;

            p.x_max = HilbertLB::max_box_sz;
            p.y_max = HilbertLB::max_box_sz;
            p.z_max = HilbertLB::max_box_sz;

            p.pack_node_index = u64_max;

            PatchData pdat;

            std::mt19937 eng(0x1111);
            std::uniform_real_distribution<f32> distpos(-1, 1);

            for (u32 part_id = 0; part_id < p.data_count; part_id++){
                pdat.pos_s.push_back({distpos(eng), distpos(eng), distpos(eng)}); //r
                pdat.U1_s.push_back(0.02f); //h
                pdat.U1_s.push_back(0.00f); //omega
                pdat.U3_s.push_back({0,0,0}); //v
                pdat.U3_s.push_back({0,0,0}); //a
                pdat.U3_s.push_back({0,0,0}); //a_old
            }
                

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
        sched.patch_data.sim_box.min_box_sim_s = {-1};
        sched.patch_data.sim_box.max_box_sim_s = {1};

        // std::cout << sched.dump_status() << std::endl;

        std::cout << "build local" << std::endl;
        sched.owned_patch_id = sched.patch_list.build_local();
        sched.patch_list.build_local_idx_map();
        sched.update_local_dtcnt_value();
        sched.update_local_load_value();

        // sched.patch_list.build_global();

        {
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
        }
    }

    static void step(SchedulerMPI &sched, TestSimInfo &siminfo) {

        SyCLHandler &hndl = SyCLHandler::get_instance();

        // std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);

        std::cout << " reduc " << std::endl;
        {

            // std::cout << sched.dump_status() << std::endl;

            // PatchField<u64> dtcnt_field;
            // dtcnt_field.local_nodes_value.resize(sched.patch_list.local.size());
            // for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
            //     dtcnt_field.local_nodes_value[idx] = sched.patch_list.local[idx].data_count;
            // }

            // std::cout << "dtcnt_field.build_global(mpi_type_u64);" << std::endl;
            // dtcnt_field.build_global(mpi_type_u64);

            // std::cout << "len 1 : " << dtcnt_field.local_nodes_value.size() << std::endl;
            // std::cout << "len 2 : " << dtcnt_field.global_values.size() << std::endl;

            SerialPatchTree<f32_3> sptree(sched.patch_tree, sched.get_box_tranform<f32_3>());
            // sptree.dump_dat();

            // std::cout << "len 3 : " << sptree.get_element_count() << std::endl;

            // std::cout << "sptree.attach_buf();" << std::endl;
            sptree.attach_buf();

            // std::cout << "sptree.reduce_field" << std::endl;
            // PatchFieldReduction<u64> pfield_reduced =
            //     sptree.reduce_field<u64, Reduce_DataCount>(hndl.get_queue_alt(0), sched, dtcnt_field);

            // std::cout << "pfield_reduced.detach_buf()" << std::endl;
            // pfield_reduced.detach_buf();
            // std::cout << " ------ > " << pfield_reduced.tree_field[0] << "\n\n\n";

            PatchField<f32> h_field;
            h_field.local_nodes_value.resize(sched.patch_list.local.size());
            for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
                h_field.local_nodes_value[idx] = 0.02f;
            }
            h_field.build_global(mpi_type_f32);

            InterfaceHandler<f32_3, f32> interface_hndl;
            interface_hndl.compute_interface_list<InterfaceSelector_SPH<f32_3, f32>>(sched, sptree, h_field);
            interface_hndl.comm_interfaces(sched);
            // interface_hndl.print_current_interf_map();

            // sched.dump_local_patches(format("patches_%d_node%d", stepi, mpi_handler::world_rank));

            // Radix_Tree<u32, f32_3> r;
            // auto& a = r.pos_min_buf;

            for (auto &[id, pdat] : sched.patch_data.owned_data) {
                if (pdat.pos_s.size() > 0) {


                    Patch &cur_p = sched.patch_list.global[sched.patch_list.id_patch_to_global_idx[id]];

                    //init acc
                    std::unique_ptr<sycl::buffer<f32_3>> pos =
                        std::make_unique<sycl::buffer<f32_3>>(pdat.pos_s.data(), pdat.pos_s.size());
                    std::unique_ptr<sycl::buffer<f32>> U1 =
                        std::make_unique<sycl::buffer<f32>>(pdat.U1_s.data(), pdat.U1_s.size());
                    std::unique_ptr<sycl::buffer<f32_3>> U3 =
                        std::make_unique<sycl::buffer<f32_3>>(pdat.U3_s.data(), pdat.U3_s.size());




                    std::unique_ptr<sycl::buffer<f32>> h_buf =
                        std::make_unique<sycl::buffer<f32>>(pdat.pos_s.size());

                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto U1acc = U1->get_access<sycl::access::mode::read>(cgh);
                        auto hacc = h_buf->get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class Modify_pos>(sycl::range(pos->size()), [=](sycl::item<1> item) {
                            u32 i = (u32)item.get_id(0);

                            hacc[i] = U1acc[i*2 + 0];
                        });
                    });


                
                    //radix tree computation
                    Radix_Tree<u32, f32_3> rtree =
                        Radix_Tree<u32, f32_3>(hndl.get_queue_compute(0), sched.patch_data.sim_box.get_box<f32>(cur_p), pos);
                    rtree.compute_cellvolume(hndl.get_queue_compute(0));
                    rtree.compute_int_boxes(hndl.get_queue_compute(0),h_buf );


                    h_buf.reset();




























                    

                    //computation kernel
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto hpart = h_buf->get_access<sycl::access::mode::read>(cgh);
                        auto r = pos->get_access<sycl::access::mode::read>(cgh);

                        walker::Radix_tree_accessor<u32, f32_3> tree_acc(rtree, cgh);

                        auto cell_int_r = rtree.buf_cell_interact_rad->get_access<sycl::access::mode::read>(cgh);

                        using Kernel = sph::kernels::M4<f32>;

                        cgh.parallel_for<class SPHTest>(sycl::range(pos->size()), [=](sycl::item<1> item) {
                            u32 id_a = (u32)item.get_id(0);

                            f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                            f32 h_a = hpart[id_a];

                            f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                            f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                            f32_3 sum_axyz{0,0,0};

                            walker::rtree_for(
                                tree_acc,
                                [&](u32 node_id) {
                                    f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                    f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                    float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                                    using namespace walker::interaction_crit;

                                    return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                               cur_pos_max_cell_b, int_r_max_cell);
                                },
                                [&](u32 id_b) {
                                    f32_3 dr = xyz_a - r[id_b];
                                    f32 rab = sycl::length(dr);
                                    f32 h_b = hpart[id_b];

                                    if(rab > h_a*Kernel::Rkern && rab > h_b*Kernel::Rkern) return;

                                    f32_3 r_ab_unit = dr / rab;

                                    if(rab < 1e-9){
                                        r_ab_unit = {0,0,0};
                                    }

                                    sum_axyz += sph_pressure(
                                        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f
                                        , 0.f,0.f, 
                                        r_ab_unit*r_ab_unit*Kernel::dW(rab,h_a), 
                                        r_ab_unit*r_ab_unit*Kernel::W(rab,h_b));

                                },
                                [](u32 node_id) {});
                        });
                    });
                }
            }

            reatribute_particles(sched, sptree);
        }
    }
};

template <class Timestepper, class SimInfo> class SimulationSPH {
  public:
    static void run_sim() {

        SchedulerMPI sched = SchedulerMPI(1e6, 1);
        sched.init_mpi_required_types();

        logfiles::open_log_files();

        SimInfo siminfo;

        auto t = timings::start_timer("init timestepper", timings::timingtype::function);
        Timestepper::init(sched, siminfo);
        t.stop();

        dump_state("step" + std::to_string(0) + "/", sched);

        timings::dump_timings("### init_step ###");

        std::cout << " ------ init sim ------" << std::endl;
        for (u32 stepi = 1; stepi < 30; stepi++) {
            std::cout << " ------ step time = " << stepi << " ------" << std::endl;
            siminfo.time = stepi;

            auto step_timer = timings::start_timer("timestepper step", timings::timingtype::function);
            Timestepper::step(sched, siminfo);
            step_timer.stop();

            dump_state("step" + std::to_string(stepi) + "/", sched);

            timings::dump_timings("### "
                                  "step" +
                                  std::to_string(stepi) + " ###");
        }

        logfiles::close_log_files();

        sched.free_mpi_required_types();
    }
};

int main(int argc, char *argv[]) {

    std::cout << shamrock_title_bar_big << std::endl;

    mpi_handler::init();

    Cmdopt &opt = Cmdopt::get_instance();
    opt.init(argc, argv, "./shamrock");

    SyCLHandler &hndl = SyCLHandler::get_instance();
    hndl.init_sycl();

    SimulationSPH<TestTimestepper, TestSimInfo>::run_sim();

    mpi_handler::close();
}