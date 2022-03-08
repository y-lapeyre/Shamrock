#include "aliases.hpp"
#include "patch/patch_field.hpp"
#include "patch/patch_reduc_tree.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "unittests/shamrocktest.hpp"
#include "patchscheduler/loadbalancing_hilbert.hpp"
#include "interfaces/interface_generator.hpp"
#include "utils/string_utils.hpp"
#include <string>
#include "interfaces/interface_selector.hpp"

class Reduce_DataCount{public:
    static u64 reduce(u64 v0,u64 v1,u64 v2,u64 v3,u64 v4,u64 v5,u64 v6,u64 v7){
        return v0+v1+v2+v3+v4+v5+v6+v7;
    }
};

Test_start("patch::patch_reduc_tree::", generation, -1){


    SyCLHandler & hndl = SyCLHandler::get_instance();

    

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


    //sched.patch_list.build_global();

    {
        PatchField<u64> dtcnt_field;
        dtcnt_field.local_nodes_value.resize(sched.patch_list.local.size());
        for(u64 idx = 0 ; idx < sched.patch_list.local.size(); idx ++){
            dtcnt_field.local_nodes_value[idx] = sched.patch_list.local[idx].data_count;
        }

        std::cout << "dtcnt_field.build_global(mpi_type_u64);" << std::endl;
        dtcnt_field.build_global(mpi_type_u64);

        std::cout << "len 1 : " << dtcnt_field.local_nodes_value.size() << std::endl;
        std::cout << "len 2 : " << dtcnt_field.global_values.size() << std::endl;

        SerialPatchTree<f32_3> sptree(sched.patch_tree, sched.get_box_tranform<f32_3>());

        std::cout << "len 3 : " << sptree.get_element_count() << std::endl;


        std::cout << "sptree.attach_buf();" << std::endl;
        sptree.attach_buf();


        

        std::cout << "sptree.reduce_field" << std::endl;
        PatchFieldReduction<u64> pfield_reduced = sptree.reduce_field<u64, Reduce_DataCount>(hndl.alt_queues[0], sched, dtcnt_field);
        
        std::cout << "pfield_reduced.detach_buf()" << std::endl;
        pfield_reduced.detach_buf();
        std::cout << " ------ > " << pfield_reduced.tree_field[0] << "\n\n\n";
    
    
        PatchField<f32> h_field;
        h_field.local_nodes_value.resize(sched.patch_list.local.size());
        for(u64 idx = 0 ; idx < sched.patch_list.local.size(); idx ++){
            h_field.local_nodes_value[idx] = 0.1f;
        }
        h_field.build_global(mpi_type_f32);

        Interface_Generator<f32_3,f32,InterfaceSelector_SPH<f32_3,f32>>::gen_interfaces_test(sched, sptree, h_field,format("interfaces_%d_node%d",0,mpi_handler::world_rank));

        

        sched.dump_local_patches(format("patches_%d_node%d",0,mpi_handler::world_rank));


    }
    





    for(u32 stepi = 1 ; stepi < 6; stepi ++){
        std::cout << " ------ step time = " <<stepi<< " ------" << std::endl;
        //std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);

        std::cout << " reduc " << std::endl;
        {

            //std::cout << sched.dump_status() << std::endl;

            PatchField<u64> dtcnt_field;
            dtcnt_field.local_nodes_value.resize(sched.patch_list.local.size());
            for(u64 idx = 0 ; idx < sched.patch_list.local.size(); idx ++){
                dtcnt_field.local_nodes_value[idx] = sched.patch_list.local[idx].data_count;
            }

            std::cout << "dtcnt_field.build_global(mpi_type_u64);" << std::endl;
            dtcnt_field.build_global(mpi_type_u64);

            // std::cout << "len 1 : " << dtcnt_field.local_nodes_value.size() << std::endl;
            // std::cout << "len 2 : " << dtcnt_field.global_values.size() << std::endl;

            SerialPatchTree<f32_3> sptree(sched.patch_tree, sched.get_box_tranform<f32_3>());
            //sptree.dump_dat();

            // std::cout << "len 3 : " << sptree.get_element_count() << std::endl;


            // std::cout << "sptree.attach_buf();" << std::endl;
            sptree.attach_buf();



            // std::cout << "sptree.reduce_field" << std::endl;
            PatchFieldReduction<u64> pfield_reduced = sptree.reduce_field<u64, Reduce_DataCount>(hndl.alt_queues[0], sched, dtcnt_field);
            
            // std::cout << "pfield_reduced.detach_buf()" << std::endl;
            pfield_reduced.detach_buf();
            std::cout << " ------ > " << pfield_reduced.tree_field[0] << "\n\n\n";


            PatchField<f32> h_field;
            h_field.local_nodes_value.resize(sched.patch_list.local.size());
            for(u64 idx = 0 ; idx < sched.patch_list.local.size(); idx ++){
                h_field.local_nodes_value[idx] = 0.1f;
            }
            h_field.build_global(mpi_type_f32);

            Interface_Generator<f32_3,f32,InterfaceSelector_SPH<f32_3,f32>>::gen_interfaces_test(sched, sptree, h_field,format("interfaces_%d_node%d",stepi,mpi_handler::world_rank));

            sched.dump_local_patches(format("patches_%d_node%d",stepi,mpi_handler::world_rank));
        }
    
        //TODO test if a interface of size 0.5x0.5x0.5 exist == error
    }

    //std::cout << sched.dump_status() << std::endl;
    
    std::cout << "changing crit\n";
    sched.crit_patch_merge = 30;
    sched.crit_patch_split = 100;
    sched.scheduler_step(true, true);


    //std::cout << sched.dump_status() << std::endl;
    //*/



    sched.free_mpi_required_types();

}