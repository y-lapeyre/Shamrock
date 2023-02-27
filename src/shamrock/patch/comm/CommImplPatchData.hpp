// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "shamrock/patch/PatchData.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclHelper.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamsys/comm/CommRequests.hpp"
#include "shamsys/comm/ProtocolEnum.hpp"
#include "shamsys/comm/details/CommImplBuffer.hpp"
#include "shamutils/throwUtils.hpp"

#include <optional>
#include <stdexcept>

namespace shamsys::comm::details {

    template<>
    class CommDetails<shamrock::patch::PatchData> {

        public:
        u32 obj_cnt;
        std::optional<u64> start_index;
        
        shamrock::patch::PatchDataLayout & pdl;

    };

    template<Protocol comm_mode>
    class CommBuffer<shamrock::patch::PatchData, comm_mode> {

        CommDetails<shamrock::patch::PatchData> details;

    };

} // namespace shamsys::comm::details