// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/patch/comm/CommImplPatchData.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamtest/shamtest.hpp"


void test_constructor_pdat_comm_buf(u64 seed, shamsys::comm::Protocol prot){

    using namespace shamrock::patch;
    using namespace shamsys::comm;

    u32 npart = 1e5;
    PatchDataLayout pdl;

    pdl.add_field<u32_3>("field1", 1);

    PatchData pdat = PatchData::mock_patchdata(seed, npart, pdl);
    CommBuffer buf {pdat, prot};
    //PatchData pdat_comp2 = CommBuffer<PatchData>::convert(std::move(buf));
    //shamtest::asserts().assert_bool("pdats matches", pdat == pdat_comp2);


}

TestStart(Unittest, "shamrock/patch/comm:CommBuffer-PatchData-ctr-destr", test_comm_buffer_patchdatactrdestr  , 1){

    test_constructor_pdat_comm_buf(0x111, shamsys::comm::DirectGPU);

}








template<class T> void test_comm_pdat_comm_buf(std::string prefix, u64 seed, shamsys::comm::Protocol prot){

    

}


template<class T> void test_comm_probe_pdat_comm_buf(std::string prefix, u64 seed, shamsys::comm::Protocol prot){

    

}


TestStart(Unittest, "shamrock/patch/comm:CommBuffer-PatchData-isend-irecv", isend_irecv_patchdata, 2){

    
}



TestStart(Unittest, "shamrock/patch/comm:CommBuffer-PatchData-isend-irecv_probe", isend_irecv_probe_patchdata, 2){

    
}