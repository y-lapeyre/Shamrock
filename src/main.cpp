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
#include <memory>
#include <mpi.h>
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
    u32 time;
};





class CurDataLayout{public:

    using pos_type = f32;

    
    
    class U1_s{public:
        static constexpr u32 nvar = 2;

        static constexpr std::array<const char*, 2> varnames {"hpart","omega"};

        static constexpr u32 ihpart = 0;
        static constexpr u32 iomega = 1;
    };

    
    class U3_s{public:
        static constexpr std::array<const char*, 3> varnames {"vxyz","axyz","axyz_old"};
        static constexpr u32 nvar = 3;

        static constexpr u32 ivxyz = 0;
        static constexpr u32 iaxyz = 1;
        static constexpr u32 iaxyz_old = 2;
    };

    // template<>
    // class U1<f64>{public:
    //     static constexpr std::array<const char*, 0> varnames {};
    //     static constexpr u32 nvar = 0;
    // };

    // template<>
    // class U3<f64>{public:
    //     static constexpr std::array<const char*, 0> varnames {};
    //     static constexpr u32 nvar = 0;
    // };

    template<class prec>
    struct U1 { using T = std::void_t<>; };
    template<class prec>
    struct U3 { using T = std::void_t<>; };
    template<>
    struct U1<f32> { using T = U1_s; };
    template<>
    struct U3<f32> { using T = U3_s; };
};


class TestTimestepper {
  public:

    

    static void init(SchedulerMPI &sched, TestSimInfo &siminfo) {

        patchdata_layout::set(1, 0, 2, 0, 3, 0);
        patchdata_layout::sync(MPI_COMM_WORLD);

        if (mpi_handler::world_rank == 0) {

            auto t = timings::start_timer("dumm setup", timings::timingtype::function);
            Patch p;

            p.data_count    = 1e6;
            p.load_value    = 1e6;
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
                pdat.pos_s.emplace_back(f32_3{distpos(eng), distpos(eng), distpos(eng)}); //r
                //                      h    omega
                pdat.U1_s.emplace_back(0.02f);
                pdat.U1_s.emplace_back(0.00f);
                //                           v          a             a_old
                pdat.U3_s.emplace_back(f32_3{0.f,0.f,0.f});
                pdat.U3_s.emplace_back(f32_3{0.f,0.f,0.f});
                pdat.U3_s.emplace_back(f32_3{0.f,0.f,0.f});
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

    static void step(SchedulerMPI &sched, TestSimInfo &siminfo) {

        SPHTimestepperLeapfrog<CurDataLayout> leapfrog;

        SyCLHandler &hndl = SyCLHandler::get_instance();

        // std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);

        leapfrog.step(sched);
    }
};

template <class Timestepper, class SimInfo> class SimulationSPH {
  public:
    static void run_sim() {

        SchedulerMPI sched = SchedulerMPI(1e5, 1);
        sched.init_mpi_required_types();

        logfiles::open_log_files();

        SimInfo siminfo;

        std::cout << " ------ init sim ------" << std::endl;

        auto t = timings::start_timer("init timestepper", timings::timingtype::function);
        Timestepper::init(sched, siminfo);
        t.stop();

        std::cout << " --- init sim done ----" << std::endl;



        std::filesystem::create_directory("step" + std::to_string(0));

        std::cout << "dumping state"<<std::endl;
        dump_state("step" + std::to_string(0) + "/", sched);

        timings::dump_timings("### init_step ###");

        
        for (u32 stepi = 1; stepi < 30; stepi++) {
            std::cout << " ------ step time = " << stepi << " ------" << std::endl;
            siminfo.time = stepi;

            auto step_timer = timings::start_timer("timestepper step", timings::timingtype::function);
            Timestepper::step(sched, siminfo);
            step_timer.stop();

            std::filesystem::create_directory("step" + std::to_string(stepi));

            dump_state("step" + std::to_string(stepi) + "/", sched);

            timings::dump_timings("### "
                                  "step" +
                                  std::to_string(stepi) + " ###");
        }

        logfiles::close_log_files();

        sched.free_mpi_required_types();
    }
};


/*
template<class T>
struct ConvertToVec{typedef std::vector<T> Type;};

template <typename... Args>
struct convert_tuple_to_vec;

template <typename... Args>
struct convert_tuple_to_vec<std::tuple<Args...>>
{
    typedef std::tuple<typename ConvertToVec<Args>::Type...> type;
};

using ttt = std::tuple<f32,f64,u32>;
using ttt_res  = std::tuple<
        std::vector<f32>,
        std::vector<f64>,
        std::vector<u32>
    >;

typedef convert_tuple_to_vec<ttt>::type tt2 ;

static_assert(
        std::is_same<
            convert_tuple_to_vec<ttt>::type,
            ttt_res
        >::value, ""
    );

*/

class DL{
    using pos = f32_3;
    using type1 = f32;
    using type2 = f32;
};

class empty_desc{};

template<
    class _Tpos,
    class _T1 = u8, u32 _nvar1 = 0,class _Desc1 = empty_desc,
    class _T2 = u8, u32 _nvar2 = 0,class _Desc2 = empty_desc,
    class _T3 = u8, u32 _nvar3 = 0,class _Desc3 = empty_desc,
    class _T4 = u8, u32 _nvar4 = 0,class _Desc4 = empty_desc,
    class _T5 = u8, u32 _nvar5 = 0,class _Desc5 = empty_desc,
    class _T6 = u8, u32 _nvar6 = 0,class _Desc6 = empty_desc,
    class _T7 = u8, u32 _nvar7 = 0,class _Desc7 = empty_desc,
    class _T8 = u8, u32 _nvar8 = 0,class _Desc8 = empty_desc
>
class Layout{public:
    using Tpos = _Tpos;

    using T1 = _T1;
    using T2 = _T2;
    using T3 = _T3;
    using T4 = _T4;
    using T5 = _T5;
    using T6 = _T6;
    using T7 = _T7;
    using T8 = _T8;


    static constexpr u32 nvar1 = _nvar1;
    static constexpr u32 nvar2 = _nvar2;
    static constexpr u32 nvar3 = _nvar3;
    static constexpr u32 nvar4 = _nvar4;
    static constexpr u32 nvar5 = _nvar5;
    static constexpr u32 nvar6 = _nvar6;
    static constexpr u32 nvar7 = _nvar7;
    static constexpr u32 nvar8 = _nvar8;

    using desc1 = _Desc1;
    using desc2 = _Desc2;
    using desc3 = _Desc3;
    using desc4 = _Desc4;
    using desc5 = _Desc5;
    using desc6 = _Desc6;
    using desc7 = _Desc7;
    using desc8 = _Desc8;
};

template<class Layout > class PData{
    std::vector<typename Layout::Tpos> pos;
    std::vector<typename Layout::T1> v1;
    std::vector<typename Layout::T2> v2;
    std::vector<typename Layout::T3> v3;
    std::vector<typename Layout::T4> v4;
    std::vector<typename Layout::T5> v5;
    std::vector<typename Layout::T6> v6;
    std::vector<typename Layout::T7> v7;
};


int main(int argc, char *argv[]) {

    class V1_layout{public:
        u32 ihpart = 0;
    };

    class V2_layout{public:
        u32 ivxyz = 0;
        u32 iaxyz = 1;
        u32 iaxyz_old = 2;
    };

    using Lay = Layout<
        f32_3, 
        f32  ,1,V1_layout,
        f32_3,3,V2_layout>;

    PData<Lay> aa;


    std::cout << shamrock_title_bar_big << std::endl;

    mpi_handler::init();

    Cmdopt &opt = Cmdopt::get_instance();
    opt.init(argc, argv, "./shamrock");

    SyCLHandler &hndl = SyCLHandler::get_instance();
    hndl.init_sycl();

    SimulationSPH<TestTimestepper, TestSimInfo>::run_sim();

    mpi_handler::close();
}