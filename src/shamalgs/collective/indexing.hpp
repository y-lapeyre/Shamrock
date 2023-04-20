// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/SyclMpiTypes.hpp"

namespace shamalgs::collective {

    struct ViewInfo{
        u64 total_byte_count;
        u64 head_offset;
    };

    
    inline ViewInfo fetch_view(u64 byte_count){
        u64 scan_val;
        mpi::exscan(&byte_count, &scan_val, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

        u64 sum_val;
        mpi::allreduce(&byte_count, &sum_val, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

        logger::raw_ln(byte_count, "->",scan_val, "sum:",sum_val);

        return {sum_val,scan_val};
    }

} // namespace shamalgs::collective