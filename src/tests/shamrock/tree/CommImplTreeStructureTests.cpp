// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/tree/TreeStructure.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/comm/ProtocolEnum.hpp"
#include "shamtest/shamtest.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamrock/tree/comm/CommImplTreeStructure.hpp"
#include <vector>

template<class u_morton>
inline void test_tree_structure_comm_template(shamsys::comm::Protocol prot){
    std::vector<u_morton> morton_list = {
        0x0,
        0x1,
        0x2,
        0x3,
        0x4,
        0x5,
        0x6,
        0x7,
        0x8,
        //0x9,
        //0xa,
        //0xb,
        0xc,
        0xd,
        0xe,
        0xf,
    };

    sycl::buffer<u_morton> buf_morton (morton_list.data(), morton_list.size());

    using namespace shamrock::tree;
    using namespace shamsys::comm;

    TreeStructure<u_morton> tstruct;

    sycl::queue & q = shamsys::instance::get_compute_queue();

    tstruct.build(
        shamsys::instance::get_compute_queue(), 
        morton_list.size()-1, 
        buf_morton
        );
    

    if(shamsys::instance::world_rank == 0){

        CommBuffer buf {tstruct, prot};
        CommRequests rqs;
        buf.isend(rqs, 1,0,MPI_COMM_WORLD);
        rqs.wait_all();

    }else if (shamsys::instance::world_rank == 1) {
        
        CommRequests rqs;
        auto buf = CommBuffer<TreeStructure<u_morton>>::irecv_probe(rqs, 0,0,MPI_COMM_WORLD,prot,{});

        rqs.wait_all();

        TreeStructure<u_morton> buf_comp2 = CommBuffer<TreeStructure<u_morton>>::convert(std::move(buf));

        shamtest::asserts().assert_bool("received correct", buf_comp2 == tstruct);
    }
}

TestStart(Unittest, "shamrock/tree/TreeStructure/comm", test_tree_structure_comm, 2){
    
    test_tree_structure_comm_template<u32>(shamsys::comm::CopyToHost);
    test_tree_structure_comm_template<u64>(shamsys::comm::CopyToHost);

}