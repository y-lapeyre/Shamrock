#include "../shamrocktest.hpp"

#include <algorithm>
#include <iterator>
#include <mpi.h>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "test_patchdata_utils.hpp"


#include "../../scheduler/mpi_scheduler.hpp"

#include "../../sys/sycl_mpi_interop.hpp"

#include "../../flags.hpp"



void make_global_local_check_vec(std::vector<Patch> & global, std::vector<Patch> & local){
    
    global.resize(mpi_handler::world_size*11);
    {
        //fill the check vector with a pseudo random int generator (seed:0x1111)
        std::mt19937 eng(0x1111);        
        std::uniform_int_distribution<u32> distu32(u32_min,u32_max);                  
        std::uniform_int_distribution<u64> distu64(u64_min,u64_max);

        u64 id_patch = 0;
        for (Patch & element : global) {
            element.id_patch      = id_patch;
            element.id_parent     = distu64(eng);
            element.id_child_r    = distu64(eng);
            element.id_child_l    = distu64(eng);
            element.x_min         = distu64(eng);
            element.y_min         = distu64(eng);
            element.z_min         = distu64(eng);
            element.x_max         = distu64(eng);
            element.y_max         = distu64(eng);
            element.z_max         = distu64(eng);
            element.data_count    = distu32(eng);
            element.node_owner_id = distu32(eng);
            element.flags         = distu32(eng) % u32(u8_max);

            id_patch++;
        }
    }



    {
        std::vector<u32> pointer_start_node(mpi_handler::world_size);
        pointer_start_node[0] = 0;
        for(i32 i = 1; i < mpi_handler::world_size; i ++){
            pointer_start_node[i] = pointer_start_node[i-1] +5+ ((i-1)%5)*((i-1)%5);
        }
        pointer_start_node.push_back(mpi_handler::world_size*11);


        for(i32 irank = 0; irank < mpi_handler::world_size; irank ++){
            for(u32 id = pointer_start_node[irank]; id < pointer_start_node[irank+1]; id ++){
                global[id].node_owner_id = irank;
            }
        }

        for(u32 id = pointer_start_node[mpi_handler::world_rank]; id < pointer_start_node[mpi_handler::world_rank+1]; id ++){
            local.push_back(global[id]);
        }
    }


}









Test_start("mpi_scheduler::",build_select_corectness,-1){
    create_MPI_patch_type();

    //in the end this vector should be recovered in recv_vec
    std::vector<Patch> check_vec;
    //divide the check_vec in local_vector on each node
    std::vector<Patch> local_check_vec;
    make_global_local_check_vec(check_vec, local_check_vec);


    MpiScheduler sche = MpiScheduler();
    for(const Patch &p : check_vec){
        sche.patch_list.global.push_back(p);
    }
    sche.patch_list.build_local();


    //check corectness of local patch list
    bool corect_size = sche.patch_list.local.size() == local_check_vec.size();
    Test_assert("corect size for local patch", corect_size);
    for(u32 i = 0 ; i < sche.patch_list.local.size(); i++){
        if(corect_size){
            Test_assert("corect patch", sche.patch_list.local[i] == local_check_vec[i]);
        }
    }


    sche.patch_list.global.clear();
    sche.patch_list.sync_global();


    corect_size = sche.patch_list.global.size() == check_vec.size();
    Test_assert("corect size for global patch", corect_size);
    for(u32 i = 0 ; i < sche.patch_list.global.size(); i++){
        if(corect_size){
            Test_assert("corect patch", sche.patch_list.global[i] == check_vec[i]);
        }
    }



    free_MPI_patch_type();
}







Test_start("mpi_scheduler::", xchg_patchs, -1){


    std::mt19937 dummy_patch_eng(0x1234);
    std::mt19937 eng(0x1111);  
    

    //in the end this vector should be recovered in recv_vec
    std::vector<Patch> check_vec;
    std::map<u64, PatchData> check_patchdata;
    //divide the check_vec in local_vector on each node
    std::vector<Patch> local_check_vec;
    make_global_local_check_vec(check_vec, local_check_vec);




    MpiScheduler sche = MpiScheduler();
    sche.init_mpi_required_types();

    patchdata_layout::set(1, 2, 1, 5, 4, 3);
    patchdata_layout::sync(MPI_COMM_WORLD);


    Timer t;
    t.start();
    //initial setup
    for(const Patch &p : check_vec){
        sche.patch_list.global.push_back(p);
        check_patchdata[p.id_patch] = gen_dummy_data(dummy_patch_eng);
    }

    sche.owned_patch_id = sche.patch_list.build_local();

    for(const u64 a : sche.owned_patch_id){
        sche.patch_data.owned_data[a] = check_patchdata[a];
    }
    t.end();



    //check corectness of local patch list
    bool corect_size = sche.patch_list.local.size() == local_check_vec.size();
    Test_assert("corect size for local patch", corect_size);
    for(u32 i = 0 ; i < sche.patch_list.local.size(); i++){
        if(corect_size){
            Test_assert("corect patch", sche.patch_list.local[i] == local_check_vec[i]);
        }
    }






    //dummy load balancing
    std::vector<std::tuple<u32, i32, i32,i32>> change_list;

    {   
        std::uniform_int_distribution<u32> distrank(0,mpi_handler::world_size-1);
        std::vector<i32> tags_it_node(mpi_handler::world_size);
        for(u32 i = 0 ; i < sche.patch_list.global.size(); i++){

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
        check_patch_data_equal(__test_result_ref, sche.patch_data.owned_data[a], check_patchdata[a]);
    }


    sche.free_mpi_required_types();

}