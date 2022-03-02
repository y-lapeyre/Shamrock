#include "CL/sycl/handler.hpp"
#include "aliases.hpp"
#include "scheduler/loadbalancing_hilbert.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_handler.hpp"
#include "scheduler/scheduler_mpi.hpp"
#include "scheduler/hilbertsfc.hpp"



int main(int argc, char *argv[]){


    std::cout << shamrock_title_bar_big << std::endl;

    mpi_handler::init();

    SyCLHandler & hndl = SyCLHandler::get_instance();
    hndl.init_sycl();

    

    SchedulerMPI sched = SchedulerMPI(10,1);
    sched.init_mpi_required_types();

    patchdata_layout::set(1, 0, 0, 0, 0, 0);
    patchdata_layout::sync(MPI_COMM_WORLD);

    if(mpi_handler::world_rank == 0){
        Patch p;

        p.data_count = 200;
        p.load_value = 200;
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
        std::uniform_real_distribution<f32> distpos(-1,1);  

        for(u32 part_id = 0 ; part_id < p.data_count ; part_id ++)
            pdat.pos_s.push_back({distpos(eng),distpos(eng),distpos(eng)});

        

        sched.add_patch(p, pdat);

        
        
    }else{
        sched.patch_list._next_patch_id ++;
    }
    
    sched.owned_patch_id = sched.patch_list.build_local();

    //std::cout << sched.dump_status() << std::endl;
    sched.patch_list.build_global();
    //std::cout << sched.dump_status() << std::endl;


    //*
    sched.patch_tree.build_from_patchtable(sched.patch_list.global, HilbertLB::max_box_sz);
    sched.patch_data.sim_box.min_box_sim_s = {-1};
    sched.patch_data.sim_box.max_box_sim_s = {1};

    
    //std::cout << sched.dump_status() << std::endl;

    std::cout << "build local" <<std::endl;
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();


    sched.patch_list.build_global();





    





    for(u32 stepi = 0 ; stepi < 5; stepi ++){
        std::cout << " ------ step time = " <<stepi<< " ------" << std::endl;
        //std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);
        
    }

    //std::cout << sched.dump_status() << std::endl;
    
    std::cout << "changing crit\n";
    sched.crit_patch_merge = 30;
    sched.crit_patch_split = 100;
    sched.scheduler_step(true, true);


    //std::cout << sched.dump_status() << std::endl;
    //*/



    sched.free_mpi_required_types();

    mpi_handler::close();

}