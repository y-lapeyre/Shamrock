// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/DistributedData.hpp"
#include "shambase/type_aliases.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shambase/stacktrace.hpp"
#include <vector>

namespace shamalgs::collective {

    struct OutComm{
        u64 receiver_id;
        std::unique_ptr<sycl::buffer<u8>> payload;
    };

    struct OutgoingCommunications{
        std::vector<OutComm> commsout;
    };

    void distributed_data_sparse_comm(
        shambase::DistributedData<OutgoingCommunications> & out_comm){

    }

}