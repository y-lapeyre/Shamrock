// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/DistributedData.hpp"
namespace shamrock::patch {

    template<class T>
    class PatchField{public:

        shambase::DistributedData<T> field_all;

        

        PatchField(shambase::DistributedData<T> && field_all) : field_all(field_all){}

    };

}