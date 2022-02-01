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

    printf("%zu %zu\n",sche.patch_list.local.size() , local_check_vec.size());


    //check corectness of local patch list
    bool corect_size = sche.patch_list.local.size() == local_check_vec.size();
    Test_assert("corect size for local patch", corect_size);
    for(u32 i = 0 ; i < sche.patch_list.local.size(); i++){
        if(corect_size){
            Test_assert("corect patch", sche.patch_list.local[i] == local_check_vec[i]);
        }
    }





    sche.patch_list.global.clear();

    create_MPI_patch_type();

    sche.patch_list.sync_global();

    free_MPI_patch_type();


    corect_size = sche.patch_list.global.size() == check_vec.size();
    Test_assert("corect size for global patch", corect_size);


    for(u32 i = 0 ; i < sche.patch_list.global.size(); i++){
        if(corect_size){
            Test_assert("corect patch", sche.patch_list.global[i] == check_vec[i]);
        }
    }


}


Test_start("mpi_scheduler::", xchg_patchs, -1){


    //in the end this vector should be recovered in recv_vec
    std::vector<Patch> check_vec;
    std::map<u64, PatchData> check_patchdata;
    //divide the check_vec in local_vector on each node
    std::vector<Patch> local_check_vec;

    make_global_local_check_vec(check_vec, local_check_vec);




    MpiScheduler sche = MpiScheduler();

    create_sycl_mpi_types();
    patchdata_layout::set(1, 2, 1, 5, 4, 3);
    patchdata_layout::sync(MPI_COMM_WORLD);


    std::mt19937 dummy_patch_eng(0x1234);
    for(const Patch &p : check_vec){
        sche.patch_list.global.push_back(p);
        check_patchdata[p.id_patch] = gen_dummy_data(dummy_patch_eng);
    }

    sche.owned_patch_id = sche.patch_list.build_local();


    std::cout << "owned : ";
    for(const u64 a : sche.owned_patch_id){
        std::cout << a << " ";
        sche.owned_patchdata[a] = check_patchdata[a];
    }std::cout << std::endl;


    std::cout << "owned patchdata : ";
    for(auto & [key,obj] : sche.owned_patchdata){
        std::cout << key << " ";
    }std::cout << std::endl;





    //check corectness of local patch list
    bool corect_size = sche.patch_list.local.size() == local_check_vec.size();
    Test_assert("corect size for local patch", corect_size);
    for(u32 i = 0 ; i < sche.patch_list.local.size(); i++){
        if(corect_size){
            Test_assert("corect patch", sche.patch_list.local[i] == local_check_vec[i]);
        }
    }



    sche.patch_list.local.clear();



    std::mt19937 eng(0x1111);  


    

          
    std::uniform_int_distribution<u32> distrank(0,mpi_handler::world_size-1);





    std::vector<std::tuple<u32, i32, i32,i32>> change_list;

    {
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



    std::vector<u64> patchdata_id_to_delete;

    std::vector<MPI_Request> rq_lst;

    for(u32 i = 0 ; i < change_list.size(); i++){
        
        auto & [idx,old_owner,new_owner,tag_comm] = change_list[i];

        //if i'm sender
        if(old_owner == mpi_handler::world_rank){
            
            auto & patchdata = sche.owned_patchdata[sche.patch_list.global[idx].id_patch];

            std::cout << "send : " << idx << " " << old_owner << " -> " << new_owner  <<  " tag : "<< tag_comm << std::endl;

            patchdata_isend(patchdata, rq_lst, new_owner, tag_comm, MPI_COMM_WORLD);
            patchdata_id_to_delete.push_back(sche.patch_list.global[old_owner].id_patch);
        }
    }



    for(u32 i = 0 ; i < change_list.size(); i++){
        auto & [idx,old_owner,new_owner,tag_comm] = change_list[i];
        auto & id_patch = sche.patch_list.global[idx].id_patch;
        

        //if i'm sender
        if(new_owner == mpi_handler::world_rank){

            std::cout << "recv : " << idx << " " << old_owner << " -> " << new_owner  <<  " tag : "<< tag_comm << std::endl;

            sche.owned_patchdata[id_patch] = patchdata_irecv( rq_lst, old_owner, tag_comm, MPI_COMM_WORLD);
        }
    }

    std::vector<MPI_Status> st_lst(rq_lst.size());
    mpi::waitall(rq_lst.size(), rq_lst.data(), st_lst.data());


    for(u32 i = 0 ; i < change_list.size(); i++){
        auto & [idx,old_owner,new_owner,tag_comm] = change_list[i];
        auto & id_patch = sche.patch_list.global[idx].id_patch;
        
        sche.patch_list.global[idx].node_owner_id = new_owner;

        //if i'm sender
        if(old_owner == mpi_handler::world_rank){
            std::cout << "deleting : " << idx << std::endl;
            sche.owned_patchdata.erase(id_patch);
        }

    }







    sche.owned_patch_id = sche.patch_list.build_local();


    std::cout << "owned : ";
    for(const u64 a : sche.owned_patch_id){
        std::cout << a << " ";
    }std::cout << std::endl;

    std::cout << "owned patchdata : ";
    std::unordered_set<u64> id_patch_from_owned_patchadata;
    for(auto & [key,obj] : sche.owned_patchdata){
        id_patch_from_owned_patchadata.insert(key);
        std::cout << key << " ";
    }std::cout << std::endl;

    std::vector<u64> diffs;

    std::set_difference(id_patch_from_owned_patchadata.begin(),id_patch_from_owned_patchadata.end(),sche.owned_patch_id.begin(),sche.owned_patch_id.end(),std::back_inserter(diffs));

    std::cout << "owned diffs cnt : " << diffs.size()<< std::endl;


    Test_assert("same id owned (patch/Data)", diffs.size() == 0);

    for(const u64 a : sche.owned_patch_id){
        check_patch_data_equal(__test_result_ref, sche.owned_patchdata[a], check_patchdata[a]);
    }



    free_sycl_mpi_types();

}