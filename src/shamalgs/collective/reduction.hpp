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
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclMpiTypes.hpp"

namespace shamalgs::collective {

    template<class T>
    inline T allreduce_sum(T a){
        T sum_val;
        mpi::allreduce(&a, &sum_val, 1, get_mpi_type<T>(), MPI_SUM, MPI_COMM_WORLD);
        return sum_val;
    }

}