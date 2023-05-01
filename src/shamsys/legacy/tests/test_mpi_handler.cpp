// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"

#include <random>
#include <vector>

#if false
Test_start("mpi::",vector_allgather_ks,-1){

    //in the end this vector should be recovered in recv_vec
    std::vector<u32> check_vec(shamsys::instance::world_size*11);
    {
        //fill the check vector with a pseudo random int generator (seed:0x1111)
        std::mt19937 eng(0x1111);                                     
        std::uniform_int_distribution<u32> dist(u32_min,u32_max);

        for (u32 & element : check_vec) {
            element = dist(eng);
        }
    }

    //divide the check_vec in local_vector on each node
    std::vector<u32> local_vector;
    {
        std::vector<u32> pointer_start_node(shamsys::instance::world_size);
        pointer_start_node[0] = 0;
        for(i32 i = 1; i < shamsys::instance::world_size; i ++){
            pointer_start_node[i] = pointer_start_node[i-1] +5+ ((i-1)%5)*((i-1)%5);
        }
        pointer_start_node.push_back(shamsys::instance::world_size*11);

        for(u32 id = pointer_start_node[shamsys::instance::world_rank]; id < pointer_start_node[shamsys::instance::world_rank+1]; id ++){
            local_vector.push_back(check_vec[id]);
        }
    }

    //receive data
    std::vector<u32> recv_vec(shamsys::instance::world_size*11);
    mpi_handler::vector_allgatherv_ks(local_vector, MPI_INT, recv_vec, MPI_INT, MPI_COMM_WORLD);

    //check correctness
    for(u32 i = 1; i < u32(shamsys::instance::world_size)*11; i ++) {
        Test_assert(format("check_vec[%d] == recv_vec[%d]",i,i), check_vec[i] == recv_vec[i]);
    }

}



Test_start("mpi::",vector_allgather,-1){

    //in the end this vector should be recovered in recv_vec
    std::vector<u32> check_vec(shamsys::instance::world_size*11);
    {
        //fill the check vector with a pseudo random int generator (seed:0x1111)
        std::mt19937 eng(0x1111);                                     
        std::uniform_int_distribution<u32> dist(u32_min,u32_max);

        for (u32 & element : check_vec) {
            element = dist(eng);
        }
    }

    //divide the check_vec in local_vector on each node
    std::vector<u32> local_vector;
    {
        std::vector<u32> pointer_start_node(shamsys::instance::world_size);
        pointer_start_node[0] = 0;
        for(i32 i = 1; i < shamsys::instance::world_size; i ++){
            pointer_start_node[i] = pointer_start_node[i-1] +5+ ((i-1)%5)*((i-1)%5);
        }
        pointer_start_node.push_back(shamsys::instance::world_size*11);

        for(u32 id = pointer_start_node[shamsys::instance::world_rank]; id < pointer_start_node[shamsys::instance::world_rank+1]; id ++){
            local_vector.push_back(check_vec[id]);
        }
    }

    //receive data
    std::vector<u32> recv_vec;
    mpi_handler::vector_allgatherv(local_vector, MPI_INT, recv_vec, MPI_INT, MPI_COMM_WORLD);

    Test_assert("vector lenght correct", recv_vec.size() == u32(shamsys::instance::world_size)*11);
    //check correctness
    for(u32 i = 1; i < u32(shamsys::instance::world_size)*11; i ++) {
        Test_assert(format("check_vec[%d] == recv_vec[%d]",i,i), check_vec[i] == recv_vec[i]);
    }

}





template<class T>
void print_vector(std::vector<T> v){
    for (const auto & a : v) {
        std::cout << a << " ";
    }
}



Test_start("mpi::",sparse_alltoall,-1){
    
    std::vector<std::vector<       u32       >> arr_send_arr_node_id(shamsys::instance::world_size);
    std::vector<std::vector<       u32       >> arr_send_arr_tag(shamsys::instance::world_size);
    std::vector<std::vector<  std::vector<float> >> arr_send_arr_data(shamsys::instance::world_size);
    MPI_Datatype exchange_datatype = MPI_FLOAT;

    {
        //fill the check vector with a pseudo random int generator (seed:0x1111)
        std::mt19937 eng(0x1111);                                     
        std::uniform_real_distribution<float> dist_flt(-1,1);
        std::uniform_int_distribution<u32> dist_send_cnt(0,10);
        std::uniform_int_distribution<u32> dist_node_id(0,shamsys::instance::world_size-1);

        for (i32 node_id = 0; node_id < shamsys::instance::world_size; node_id++) {
            u32 send_cnt = dist_send_cnt(eng);

            arr_send_arr_node_id[node_id].resize(send_cnt);
            arr_send_arr_tag[node_id].resize(send_cnt);
            arr_send_arr_data[node_id].resize(send_cnt);
            
            for(u32 i = 0; i < send_cnt; i ++){
                arr_send_arr_node_id[node_id][i] = dist_node_id(eng);
                arr_send_arr_tag    [node_id][i] = i;

                u32 send_data_cnt = dist_send_cnt(eng);
                arr_send_arr_data   [node_id][i].resize(send_data_cnt);

                for(u32 j = 0; j < send_data_cnt; j ++){
                    arr_send_arr_data[node_id][i][j] = dist_flt(eng);
                }

            }
        }
    }


    std::vector<       u32       > recv_arr_node_id;
    std::vector<       u32       > recv_arr_tag;
    std::vector<  std::vector<float> > recv_arr_data;


    for(u32 i = 0; i < arr_send_arr_node_id[shamsys::instance::world_rank].size(); i ++){
        printf("sending to node %d : tag = %d data = ",arr_send_arr_node_id[shamsys::instance::world_rank][i],arr_send_arr_tag    [shamsys::instance::world_rank][i]);
        print_vector(arr_send_arr_data   [shamsys::instance::world_rank][i]);
        printf("\n");
    }

    mpi_handler::sparse_alltoall(
        arr_send_arr_node_id[shamsys::instance::world_rank], 
        arr_send_arr_tag[shamsys::instance::world_rank], 
        arr_send_arr_data[shamsys::instance::world_rank], 

        exchange_datatype, 
        
        recv_arr_node_id, 
        recv_arr_tag, 
        recv_arr_data, 
        
        shamsys::instance::world_size, 
        MPI_COMM_WORLD);

    for(u32 i = 0; i < recv_arr_node_id.size(); i ++){
        printf("received from to node %d : tag = %d data = ",recv_arr_node_id[i],recv_arr_tag[i]);
        print_vector(recv_arr_data[i]);
        printf("\n");
    }

    for(u32 recv_id = 0; recv_id < recv_arr_node_id.size(); recv_id++){

        bool acc = false;

        u32 recv_node_id = recv_arr_node_id[recv_id];
        u32 recv_node_tag = recv_arr_tag[recv_id];
        std::vector<float> recv_node_data = recv_arr_data[recv_id];

        for(i32 sender_rank = 0; sender_rank < shamsys::instance::world_size; sender_rank ++){
            for(u32 send_obj_id = 0; send_obj_id < arr_send_arr_node_id[sender_rank].size(); send_obj_id ++){

                u32 send_node_id = sender_rank;
                u32 send_node_tag = arr_send_arr_tag[sender_rank][send_obj_id];
                std::vector<float> send_node_data = arr_send_arr_data[sender_rank][send_obj_id];

                bool is_same = (recv_node_id == send_node_id) && (recv_node_tag == send_node_tag) && (recv_node_data == send_node_data);
                acc = acc || is_same;
            }
        }
        
        Test_assert(format("object %d match",recv_id), acc);
    }
    
}

#endif