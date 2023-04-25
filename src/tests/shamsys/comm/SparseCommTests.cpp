// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random/random.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shamsys/comm/SparseCommunicator.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include "shamsys/comm/details/CommImplBuffer.hpp"

template<class T>
void test_sparse_comm(std::string prefix, u64 seed, shamsys::comm::Protocol prot){

    //logger::raw_ln(prefix);

    const u32 wsize = shamsys::instance::world_size;
    const u32 wrank = shamsys::instance::world_rank;

    u32 num_buf = wsize*5;
    u32 npart_per_buf = 1e4;

    using namespace shamsys::comm;

    
    std::vector<u32> owner_rank;

    SparseCommSource<sycl::buffer<T>> send_bufs;

    std::mt19937 eng(seed);

    for(u32 i = 0; i < num_buf; i++){
        owner_rank.push_back(shamalgs::random::mock_value<u32>(eng, 0,wsize-1));
        send_bufs.push_back(
            std::make_unique<sycl::buffer<T>>(shamalgs::random::mock_buffer<T>(seed, npart_per_buf))
        );
    }

    SparseCommSourceBuffers<sycl::buffer<T>> send_comm_bufs;

    for(u32 i = 0; i < num_buf; i++){
        send_comm_bufs.push_back(
            std::make_unique<CommBuffer<sycl::buffer<T>>>(
                *send_bufs[i], prot
            )
        );
    }


    std::vector<u64_2> comm_vec_local;

    for(u32 i = 0; i < num_buf; i++){
        if(owner_rank[i] == wrank){
            comm_vec_local.push_back({u64(i), shamalgs::random::mock_value<u64>(eng, 0,num_buf-1)});
        }
    }


    SparseGroupCommunicator scomm(comm_vec_local);

    SparseCommResultBuffers<sycl::buffer<T>> res = scomm.sparse_exchange(
        send_comm_bufs, 
        {}, 
        prot, 
        [&](u64 id){
            return owner_rank[id];
        }, [&](u64 id){
            return id;
        }
    );

    //for (u64_2 a : comm_vec_local) {
    //    logger::raw_ln(a);
    //}
    //logger::raw_ln("--------");
    //for (u64_2 a : scomm.global_comm_vec) {
    //    logger::raw_ln(a);
    //}

    //checking

    auto find_idx_comm = [&](u64 id_sender, u64 id_receiver){
        u32 idx = 0;
        for (u64_2 vec : scomm.global_comm_vec) {
            if(id_sender == vec.x() && id_receiver == vec.y()){
                return idx;
            }
            idx ++;
        }

        throw shambase::throw_with_loc<std::runtime_error>(
            shambase::format("unable to find the corresponding comm {} -> {}",id_sender,id_receiver)
            );
        return u32(0);
    };

    for(auto & [id_receiver, buf_ptr_vec] : res){
        for(auto & [id_sender, buf_ptr] : buf_ptr_vec){

            sycl::buffer<T> recv_buf = buf_ptr->copy_back();

            find_idx_comm(id_sender,id_receiver);

            sycl::buffer<T> & sender_buf = *send_bufs[id_sender];


            {
                sycl::host_accessor acc1 {recv_buf};
                sycl::host_accessor acc2 {sender_buf};

                std::string id_err_list = "errors in id : ";

                bool eq = true;
                for(u32 i = 0; i < npart_per_buf; i++){
                    if(!shambase::vec_equals(acc1[i] , acc2[i])){
                        eq = false;
                        //id_err_list += std::to_string(i) + " ";
                    }
                }

                if (eq) {
                    shamtest::asserts().assert_bool("same content", eq);
                }else{
                    shamtest::asserts().assert_add_comment("same content", eq, id_err_list);
                }
            }


        }
    }



}


TestStart(Unittest, "shamsys/comm/SparseCommunicator", testsparsecomm, -1){
    
    u64 seed = 0x111;

    using namespace shamsys::comm;
 
    test_sparse_comm<f32   >("f32   : ",seed,CopyToHost);
    test_sparse_comm<f32_2 >("f32_2 : ",seed,CopyToHost);
    test_sparse_comm<f32_3 >("f32_3 : ",seed,CopyToHost);
    test_sparse_comm<f32_4 >("f32_4 : ",seed,CopyToHost);
    test_sparse_comm<f32_8 >("f32_8 : ",seed,CopyToHost);
    test_sparse_comm<f32_16>("f32_16: ",seed,CopyToHost);
    test_sparse_comm<f64   >("f64   : ",seed,CopyToHost);
    test_sparse_comm<f64_2 >("f64_2 : ",seed,CopyToHost);
    test_sparse_comm<f64_3 >("f64_3 : ",seed,CopyToHost);
    test_sparse_comm<f64_4 >("f64_4 : ",seed,CopyToHost);
    test_sparse_comm<f64_8 >("f64_8 : ",seed,CopyToHost);
    test_sparse_comm<f64_16>("f64_16: ",seed,CopyToHost);
    test_sparse_comm<u32   >("u32   : ",seed,CopyToHost);
    test_sparse_comm<u64   >("u64   : ",seed,CopyToHost);


    test_sparse_comm<f32   >("f32   : ",seed,DirectGPU);
    test_sparse_comm<f32_2 >("f32_2 : ",seed,DirectGPU);
    test_sparse_comm<f32_3 >("f32_3 : ",seed,DirectGPU);
    test_sparse_comm<f32_4 >("f32_4 : ",seed,DirectGPU);
    test_sparse_comm<f32_8 >("f32_8 : ",seed,DirectGPU);
    test_sparse_comm<f32_16>("f32_16: ",seed,DirectGPU);
    test_sparse_comm<f64   >("f64   : ",seed,DirectGPU);
    test_sparse_comm<f64_2 >("f64_2 : ",seed,DirectGPU);
    test_sparse_comm<f64_3 >("f64_3 : ",seed,DirectGPU);
    test_sparse_comm<f64_4 >("f64_4 : ",seed,DirectGPU);
    test_sparse_comm<f64_8 >("f64_8 : ",seed,DirectGPU);
    test_sparse_comm<f64_16>("f64_16: ",seed,DirectGPU);
    test_sparse_comm<u32   >("u32   : ",seed,DirectGPU);
    test_sparse_comm<u64   >("u64   : ",seed,DirectGPU);

    test_sparse_comm<f32   >("f32   : ",seed,DirectGPUFlatten);
    test_sparse_comm<f32_2 >("f32_2 : ",seed,DirectGPUFlatten);
    test_sparse_comm<f32_3 >("f32_3 : ",seed,DirectGPUFlatten);
    test_sparse_comm<f32_4 >("f32_4 : ",seed,DirectGPUFlatten);
    test_sparse_comm<f32_8 >("f32_8 : ",seed,DirectGPUFlatten);
    test_sparse_comm<f32_16>("f32_16: ",seed,DirectGPUFlatten);
    test_sparse_comm<f64   >("f64   : ",seed,DirectGPUFlatten);
    test_sparse_comm<f64_2 >("f64_2 : ",seed,DirectGPUFlatten);
    test_sparse_comm<f64_3 >("f64_3 : ",seed,DirectGPUFlatten);
    test_sparse_comm<f64_4 >("f64_4 : ",seed,DirectGPUFlatten);
    test_sparse_comm<f64_8 >("f64_8 : ",seed,DirectGPUFlatten);
    test_sparse_comm<f64_16>("f64_16: ",seed,DirectGPUFlatten);
    test_sparse_comm<u32   >("u32   : ",seed,DirectGPUFlatten);
    test_sparse_comm<u64   >("u64   : ",seed,DirectGPUFlatten);
    

}
