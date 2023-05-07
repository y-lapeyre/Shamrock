// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/collective/sparseXchg.hpp"
#include "shambase/DistributedData.hpp"
#include "shambase/type_aliases.hpp"
#include <optional>
#include <stdexcept>

namespace shamalgs::collective {


    using SerializedDDataComm = shambase::DistributedDataShared<std::unique_ptr<sycl::buffer<u8>>>;

    void distributed_data_sparse_comm(SerializedDDataComm &send_distrib_data,
                                             SerializedDDataComm &recv_distrib_data,
                                             shamsys::CommunicationProtocol prot,
                                             std::function<i32(u64)> rank_getter,
                                             std::optional<SparseCommTable> comm_table = {});

} // namespace shamalgs::collective