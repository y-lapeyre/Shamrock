// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/comm/CommImplPatchDataField.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamtest/shamtest.hpp"


template<class T> void test_constructor_syclbuf(std::string prefix, u64 seed, shamsys::comm::Protocol prot){


    u32 npart = 1e5;

    PatchDataField<T> field = PatchDataField<T>::mock_field(seed, npart, "test", 1);

    using namespace shamsys::comm;

    CommBuffer buf {field, prot};

    PatchDataField<T> field_comp2 = CommBuffer<PatchDataField<T>>::convert(std::move(buf));

    shamtest::asserts().assert_bool("fields matches", field.check_field_match(field_comp2));

}

TestStart(Unittest, "shamrock/patch/comm:CommBuffer-PdatField-ctr-destr", test_comm_buffer_pdatfieldctrdestr  , 1){

    
    u64 seed = 0x111;

    using namespace shamsys::comm;
 
    test_constructor_syclbuf<f32   >("f32   : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",seed,CopyToHost);
    test_constructor_syclbuf<f32_16>("f32_16: ",seed,CopyToHost);
    test_constructor_syclbuf<f64   >("f64   : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",seed,CopyToHost);
    test_constructor_syclbuf<f64_16>("f64_16: ",seed,CopyToHost);
    test_constructor_syclbuf<u32   >("u32   : ",seed,CopyToHost);
    test_constructor_syclbuf<u64   >("u64   : ",seed,CopyToHost);


    test_constructor_syclbuf<f32   >("f32   : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPU);
    test_constructor_syclbuf<f32_16>("f32_16: ",seed,DirectGPU);
    test_constructor_syclbuf<f64   >("f64   : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPU);
    test_constructor_syclbuf<f64_16>("f64_16: ",seed,DirectGPU);
    test_constructor_syclbuf<u32   >("u32   : ",seed,DirectGPU);
    test_constructor_syclbuf<u64   >("u64   : ",seed,DirectGPU);

    test_constructor_syclbuf<f32   >("f32   : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f32_16>("f32_16: ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64   >("f64   : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<f64_16>("f64_16: ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<u32   >("u32   : ",seed,DirectGPUFlatten);
    test_constructor_syclbuf<u64   >("u64   : ",seed,DirectGPUFlatten);

}








template<class T> void test_comm_syclbuf(std::string prefix, u64 seed, shamsys::comm::Protocol prot){

    
    u32 npart = 1e4;

    PatchDataField<T> field_comp = PatchDataField<T>::mock_field(seed, npart, "test", 1);

    using namespace shamsys::comm;



    if(shamsys::instance::world_rank == 0){

        CommBuffer buf {field_comp, prot};


        CommRequests rqs;
        buf.isend(rqs, 1,0,MPI_COMM_WORLD);

        rqs.wait_all();

    }else if (shamsys::instance::world_rank == 1) {
        CommDetails<PatchDataField<T>> det{
            npart, "test", 1, {}
        };

        CommBuffer buf {det,prot};

        CommRequests rqs;
        buf.irecv(rqs, 0,0,MPI_COMM_WORLD);
        rqs.wait_all();



        PatchDataField<T> field_comp2 = CommBuffer<PatchDataField<T>>::convert(std::move(buf));

        shamtest::asserts().assert_bool("fields matches", field_comp.check_field_match(field_comp2));
    }

}


template<class T> void test_comm_probe_syclbuf(std::string prefix, u64 seed, shamsys::comm::Protocol prot){

    
    u32 npart = 1e4;

    PatchDataField<T> field_comp = PatchDataField<T>::mock_field(seed, npart, "test", 1);

    using namespace shamsys::comm;



    if(shamsys::instance::world_rank == 0){

        CommBuffer buf {field_comp, prot};


        CommRequests rqs;
        buf.isend(rqs, 1,0,MPI_COMM_WORLD);

        rqs.wait_all();

    }else if (shamsys::instance::world_rank == 1) {

        CommDetails<PatchDataField<T>> det{
            0, "test", 1, {}
        };
        
        CommRequests rqs;
        auto buf = CommBuffer<PatchDataField<T>>::irecv_probe(rqs, 0,0,MPI_COMM_WORLD,prot,det);

        rqs.wait_all();



        PatchDataField<T> field_comp2 = CommBuffer<PatchDataField<T>>::convert(std::move(buf));

        shamtest::asserts().assert_bool("fields matches", field_comp.check_field_match(field_comp2));
    }

}


TestStart(Unittest, "shamrock/patch/comm:CommBuffer-PdatField-isend-irecv", isend_irecv_pdatfield, 2){

    u64 seed = 0x111;

    using namespace shamsys::comm;
 
    test_comm_syclbuf<f32   >("f32   : ",seed,CopyToHost);
    test_comm_syclbuf<f32_2 >("f32_2 : ",seed,CopyToHost);
    test_comm_syclbuf<f32_3 >("f32_3 : ",seed,CopyToHost);
    test_comm_syclbuf<f32_4 >("f32_4 : ",seed,CopyToHost);
    test_comm_syclbuf<f32_8 >("f32_8 : ",seed,CopyToHost);
    test_comm_syclbuf<f32_16>("f32_16: ",seed,CopyToHost);
    test_comm_syclbuf<f64   >("f64   : ",seed,CopyToHost);
    test_comm_syclbuf<f64_2 >("f64_2 : ",seed,CopyToHost);
    test_comm_syclbuf<f64_3 >("f64_3 : ",seed,CopyToHost);
    test_comm_syclbuf<f64_4 >("f64_4 : ",seed,CopyToHost);
    test_comm_syclbuf<f64_8 >("f64_8 : ",seed,CopyToHost);
    test_comm_syclbuf<f64_16>("f64_16: ",seed,CopyToHost);
    test_comm_syclbuf<u32   >("u32   : ",seed,CopyToHost);
    test_comm_syclbuf<u64   >("u64   : ",seed,CopyToHost);


    test_comm_syclbuf<f32   >("f32   : ",seed,DirectGPU);
    test_comm_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPU);
    test_comm_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPU);
    test_comm_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPU);
    test_comm_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPU);
    test_comm_syclbuf<f32_16>("f32_16: ",seed,DirectGPU);
    test_comm_syclbuf<f64   >("f64   : ",seed,DirectGPU);
    test_comm_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPU);
    test_comm_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPU);
    test_comm_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPU);
    test_comm_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPU);
    test_comm_syclbuf<f64_16>("f64_16: ",seed,DirectGPU);
    test_comm_syclbuf<u32   >("u32   : ",seed,DirectGPU);
    test_comm_syclbuf<u64   >("u64   : ",seed,DirectGPU);

    test_comm_syclbuf<f32   >("f32   : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f32_16>("f32_16: ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64   >("f64   : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<f64_16>("f64_16: ",seed,DirectGPUFlatten);
    test_comm_syclbuf<u32   >("u32   : ",seed,DirectGPUFlatten);
    test_comm_syclbuf<u64   >("u64   : ",seed,DirectGPUFlatten);
    
}



TestStart(Unittest, "shamrock/patch/comm:CommBuffer-PdatField-isend-irecv_probe", isend_irecv_probe_pdatfield, 2){

    u64 seed = 0x111;

    using namespace shamsys::comm;
 
    test_comm_probe_syclbuf<f32   >("f32   : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_2 >("f32_2 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_3 >("f32_3 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_4 >("f32_4 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_8 >("f32_8 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f32_16>("f32_16: ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64   >("f64   : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_2 >("f64_2 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_3 >("f64_3 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_4 >("f64_4 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_8 >("f64_8 : ",seed,CopyToHost);
    test_comm_probe_syclbuf<f64_16>("f64_16: ",seed,CopyToHost);
    test_comm_probe_syclbuf<u32   >("u32   : ",seed,CopyToHost);
    test_comm_probe_syclbuf<u64   >("u64   : ",seed,CopyToHost);


    test_comm_probe_syclbuf<f32   >("f32   : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f32_16>("f32_16: ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64   >("f64   : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPU);
    test_comm_probe_syclbuf<f64_16>("f64_16: ",seed,DirectGPU);
    test_comm_probe_syclbuf<u32   >("u32   : ",seed,DirectGPU);
    test_comm_probe_syclbuf<u64   >("u64   : ",seed,DirectGPU);

    test_comm_probe_syclbuf<f32   >("f32   : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_2 >("f32_2 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_3 >("f32_3 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_4 >("f32_4 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_8 >("f32_8 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f32_16>("f32_16: ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64   >("f64   : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_2 >("f64_2 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_3 >("f64_3 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_4 >("f64_4 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_8 >("f64_8 : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<f64_16>("f64_16: ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<u32   >("u32   : ",seed,DirectGPUFlatten);
    test_comm_probe_syclbuf<u64   >("u64   : ",seed,DirectGPUFlatten);
    
}