#include "../shamrocktest.hpp"

#include <random>
#include <vector>

Test_start("mpi::",vector_allgather_ks,-1){

    //in the end this vector should be recovered in recv_vec
    std::vector<u32> check_vec(mpi_handler::world_size*11);
    {
        //fill the check vector with a pseudo random int generator (seed:0x1111)
        std::mt19937 eng(0x1111);                                     
        std::uniform_int_distribution<u32> dist(0,4294967295);

        for (u32 & element : check_vec) {
            element = dist(eng);
        }
    }

    //divide the check_vec in local_vector on each node
    std::vector<u32> local_vector;
    {
        std::vector<u32> pointer_start_node(mpi_handler::world_size);
        pointer_start_node[0] = 0;
        for(u32 i = 1; i < mpi_handler::world_size; i ++){
            pointer_start_node[i] = pointer_start_node[i-1] +5+ ((i-1)%5)*((i-1)%5);
        }
        pointer_start_node.push_back(mpi_handler::world_size*11);

        for(u32 id = pointer_start_node[mpi_handler::world_rank]; id < pointer_start_node[mpi_handler::world_rank+1]; id ++){
            local_vector.push_back(check_vec[id]);
        }
    }

    //receive data
    std::vector<u32> recv_vec(mpi_handler::world_size*11);
    mpi_handler::vector_allgatherv_ks(local_vector, MPI_INT, recv_vec, MPI_INT, MPI_COMM_WORLD);

    //check correctness
    for(u32 i = 1; i < mpi_handler::world_size*11; i ++) {
        Test_assert(format("check_vec[%d] == recv_vec[%d]",i,i), check_vec[i] == recv_vec[i]);
    }

}