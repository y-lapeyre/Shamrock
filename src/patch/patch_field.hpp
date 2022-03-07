#pragma once

#include <vector>

#include "sys/mpi_handler.hpp"

/**
 * @brief Define a field attached to a patch (exemple: FMM multipoles, hmax in SPH)
 * 
 * @tparam type type of object to store
 */
template<class type>
class PatchField{public:

    std::vector<type> local_nodes_value;

    std::vector<type> global_values;

    inline void build_global(MPI_Datatype dtype){
        mpi_handler::vector_allgatherv(local_nodes_value, dtype, global_values, dtype, MPI_COMM_WORLD);
    }

};