#include "../shamrocktest.hpp"

#include <random>
#include <vector>

#include "../../scheduler/mpi_scheduler.hpp"



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