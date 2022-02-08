#pragma once

#include <vector>
#include <random>

#include "../../scheduler/scheduler_mpi.hpp"


inline void make_global_local_check_vec(std::vector<Patch> & global, std::vector<Patch> & local){
    
    global.resize(mpi_handler::world_size*11);
    {
        //fill the check vector with a pseudo random int generator (seed:0x1111)
        std::mt19937 eng(0x1111);        
        std::uniform_int_distribution<u32> distu32(u32_min,u32_max);                  
        std::uniform_int_distribution<u64> distu64(u64_min,u64_max);

        std::uniform_int_distribution<u32> distdtcnt(0,10000); 

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
            element.data_count    = distdtcnt(eng);
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
