// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include "shambase/DistributedData.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/BasicGas.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"

namespace shammodels::sph {

    template<class vec>
    class BasicGasPeriodicGhostHandler{
        using flt      = shambase::VecComponent<vec>;

        PatchScheduler & sched;
        BasicGasPeriodicGhostHandler(PatchScheduler & sched) : sched(sched){}

        struct InterfaceGeneInfos{
            shammath::CoordRange<vec> cut_volume;
        };

        using GeneratorMap = shambase::DistributedDataShared<InterfaceGeneInfos>;


        GeneratorMap find_interfaces(shambase::DistributedData<flt> & int_range_max){
            using namespace shamrock::patch;

            for(i32 xoff = - 1; xoff <= 1; xoff ++){

            }
            sched.for_each_global_patch([&](const Patch p){
                sched.for_each_global_patch([](const Patch p){
                    
                });
            });
        }

    };
        

}