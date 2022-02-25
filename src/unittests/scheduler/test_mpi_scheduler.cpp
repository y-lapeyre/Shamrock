#include "../shamrocktest.hpp"

#include <algorithm>
#include <iterator>
#include <mpi.h>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>



#include "../../scheduler/scheduler_mpi.hpp"

#include "../../sys/sycl_mpi_interop.hpp"

#include "../../flags.hpp"



#include "test_patch_utils.hpp"

#include "../../scheduler/hilbertsfc.hpp"






Test_start("SchedulerPatchData::", apply_change_list, -1){


    std::mt19937 dummy_patch_eng(0x1234);
    std::mt19937 eng(0x1111);  
    

    //in the end this vector should be recovered in recv_vec
    std::vector<Patch> check_vec;
    std::map<u64, PatchData> check_patchdata;
    //divide the check_vec in local_vector on each node
    std::vector<Patch> local_check_vec;
    make_global_local_check_vec(check_vec, local_check_vec);




    SchedulerMPI sche = SchedulerMPI(-1,-1);
    sche.init_mpi_required_types();

    patchdata_layout::set(1, 2, 1, 5, 4, 3);
    patchdata_layout::sync(MPI_COMM_WORLD);



    //initial setup
    for(const Patch &p : check_vec){
        sche.patch_list.global.push_back(p);
        check_patchdata[p.id_patch] = patchdata_gen_dummy_data(dummy_patch_eng);
    }

    sche.owned_patch_id = sche.patch_list.build_local();

    for(const u64 a : sche.owned_patch_id){
        sche.patch_data.owned_data[a] = check_patchdata[a];
    }
    



    //check corectness of local patch list
    bool corect_size = sche.patch_list.local.size() == local_check_vec.size();
    Test_assert("corect size for local patch", corect_size);
    for(u32 i = 0 ; i < sche.patch_list.local.size(); i++){
        if(corect_size){
            Test_assert("corect patch", sche.patch_list.local[i] == local_check_vec[i]);
        }
    }






    //dummy load balancing
    std::vector<std::tuple<u64, i32, i32,i32>> change_list;

    {   
        std::uniform_int_distribution<u32> distrank(0,mpi_handler::world_size-1);
        std::vector<i32> tags_it_node(mpi_handler::world_size);
        for(u64 i = 0 ; i < sche.patch_list.global.size(); i++){

            i32 old_owner = sche.patch_list.global[i].node_owner_id;
            i32 new_owner = distrank(eng);

            if(new_owner != old_owner){
                change_list.push_back({i,old_owner,new_owner,tags_it_node[old_owner]});
                tags_it_node[old_owner] ++;
            }
            
        }
        
    }


    //exchange data
    sche.patch_data.apply_change_list(change_list, sche.patch_list);


    //rebuild local table
    sche.owned_patch_id = sche.patch_list.build_local();


    //check for mismatch
    std::vector<u64> diffs;

    std::unordered_set<u64> id_patch_from_owned_patchadata;
    for(auto & [key,obj] : sche.patch_data.owned_data){
        id_patch_from_owned_patchadata.insert(key);
    }
    std::set_difference(id_patch_from_owned_patchadata.begin(),id_patch_from_owned_patchadata.end(),sche.owned_patch_id.begin(),sche.owned_patch_id.end(),std::back_inserter(diffs));
    Test_assert("same id owned (patch/Data)", diffs.size() == 0);

    //check corectness of patchdata contents
    for(const u64 a : sche.owned_patch_id){
        Test_assert("match data", check_patch_data_match( sche.patch_data.owned_data[a], check_patchdata[a])) ;
    }


    sche.free_mpi_required_types();

}



Test_start("mpi_scheduler::", testLB, -1){

    std::mt19937 dummy_patch_eng(0x1234);
    std::mt19937 eng(0x1111);  
    

    //in the end this vector should be recovered in recv_vec
    std::vector<Patch> check_vec;
    std::map<u64, PatchData> check_patchdata;
    //divide the check_vec in local_vector on each node
    std::vector<Patch> local_check_vec;
    make_global_local_check_vec(check_vec, local_check_vec);




    


    SchedulerMPI sche = SchedulerMPI(-1,-1);
    sche.init_mpi_required_types();

    patchdata_layout::set(1, 2, 1, 5, 4, 3);
    patchdata_layout::sync(MPI_COMM_WORLD);



    //initial setup
    for(const Patch &p : check_vec){
        sche.patch_list.global.push_back(p);
        check_patchdata[p.id_patch] = patchdata_gen_dummy_data(dummy_patch_eng);
    }

    sche.owned_patch_id = sche.patch_list.build_local();

    for(const u64 a : sche.owned_patch_id){
        sche.patch_data.owned_data[a] = check_patchdata[a];
    }
    



    //check corectness of local patch list
    bool corect_size = sche.patch_list.local.size() == local_check_vec.size();
    Test_assert("corect size for local patch", corect_size);
    for(u32 i = 0 ; i < sche.patch_list.local.size(); i++){
        if(corect_size){
            Test_assert("corect patch", sche.patch_list.local[i] == local_check_vec[i]);
        }
    }




    sche.sync_build_LB(false, true);


    


    //check for mismatch
    std::vector<u64> diffs;

    std::unordered_set<u64> id_patch_from_owned_patchadata;
    for(auto & [key,obj] : sche.patch_data.owned_data){
        id_patch_from_owned_patchadata.insert(key);
    }
    std::set_difference(id_patch_from_owned_patchadata.begin(),id_patch_from_owned_patchadata.end(),sche.owned_patch_id.begin(),sche.owned_patch_id.end(),std::back_inserter(diffs));
    Test_assert("same id owned (patch/Data)", diffs.size() == 0);

    //check corectness of patchdata contents
    for(const u64 a : sche.owned_patch_id){
         Test_assert("match data", check_patch_data_match( sche.patch_data.owned_data[a], check_patchdata[a])) ;
    }


    sche.free_mpi_required_types();

}



Test_start("mpi_scheduler::", test_split, -1){


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

        p.x_max = hilbert_box21_sz;
        p.y_max = hilbert_box21_sz;
        p.z_max = hilbert_box21_sz;

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

    std::cout << sched.dump_status() << std::endl;
    sched.patch_list.sync_global();
    std::cout << sched.dump_status() << std::endl;


    //*
    sched.patch_tree.build_from_patchtable(sched.patch_list.global, hilbert_box21_sz);
    sched.patch_data.sim_box.min_box_sim_s = {-1};
    sched.patch_data.sim_box.max_box_sim_s = {1};

    
    std::cout << sched.dump_status() << std::endl;

    std::cout << "build local" <<std::endl;
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();


    for(u32 stepi = 0 ; stepi < 5; stepi ++){
        std::cout << "step : " << std::endl;
        std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);
        
    }

    std::cout << sched.dump_status() << std::endl;
    
    std::cout << "changing crit\n";
    sched.crit_patch_merge = 30;
    sched.crit_patch_split = 100;
    sched.scheduler_step(true, true);


    std::cout << sched.dump_status() << std::endl;
    //*/



    sched.free_mpi_required_types();

}