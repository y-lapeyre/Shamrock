
#include "../unit_test_handler.hpp"
#include "../../tree/patch.hpp"
#include "../../sys/mpi_handler.hpp"

#include "../../com/sparse_mpi.hpp"

#include <cstdio>
#include <mpi.h>
#include <vector>



inline void run_tests_patch(){

    if(unit_test::test_start("tree/patch.hpp (Patch data (se/dese)rialization)", false)){

        PatchData data_clean;

        data_clean.r.push_back({0,0,0});
        data_clean.r.push_back({0,0.1,0});
        data_clean.r.push_back({0,0,0});
        data_clean.r.push_back({0.3,0,0});
        data_clean.r.push_back({0,0,0.9});
        data_clean.r.push_back({0.7,0.8,0.9});

        data_clean.obj_cnt = data_clean.r.size();

        std::vector<u8> data_ser = data_clean.serialize();

        printf("%zu\n",data_ser.size());

        printf("%d\n",* ((u32*)data_ser.data()));

        
        PatchData data_tran(data_ser);

        unit_test::test_assert("same obj count", data_clean.obj_cnt, data_tran.obj_cnt);

        for(u32 i = 0; i < data_clean.obj_cnt; i++){
            bool same = 
                (data_clean.r[i].x() == data_tran.r[i].x()) &&
                (data_clean.r[i].y() == data_tran.r[i].y()) &&
                (data_clean.r[i].z() == data_tran.r[i].z())
            ;
            unit_test::test_assert("same r", same);
        }
        

    }unit_test::test_end();




    if(unit_test::test_start("tree/patch.hpp (Sparse collective communication)", true)){


        std::vector<u32> send_loc_cnt;
        std::vector<MPI_Packet> payload_send_queue;


        switch (world_rank) {
        case 0: 
            send_loc_cnt = {1,1,0,0}; 
            payload_send_queue.push_back({ 0, {0,4,1,2,3}});
            payload_send_queue.push_back({ 1, {7,4,1,2,3}});
            break;
        case 1: 
            send_loc_cnt = {0,1,2,0}; 
            payload_send_queue.push_back({ 1, {0,4,1,2,3,7,9}});
            payload_send_queue.push_back({ 2, {0,4,8,2,3}});
            payload_send_queue.push_back({ 2, {0,4,1,2,3}});
            break;
        case 2: 
            send_loc_cnt = {1,1,0,0}; 
            payload_send_queue.push_back({ 0, {0,4,1,2,3}});
            payload_send_queue.push_back({ 1, {0,4,1,2,3,8,4}});
            break;
        case 3: 
            send_loc_cnt = {0,1,0,3}; 
            payload_send_queue.push_back({ 1, {0,4,1,2,3}});
            payload_send_queue.push_back({ 3, {0,4,1,2,3}});
            payload_send_queue.push_back({ 3, {0,4,1,2,3}});
            payload_send_queue.push_back({ 3, {0,4,1,2,3}});
            break;
        }











        

        std::vector<MPI_Request> requests(payload_send_queue.size());

        for(u32 i = 0; i < payload_send_queue.size();i++){
            //printf("async send rank = %d n°%d\n",world_rank,i);
            MPI_Isend(
                payload_send_queue[i].patchdata.data(), 
                payload_send_queue[i].patchdata.size(), 
                MPI_CHAR, 
                payload_send_queue[i].rank, 
                0, 
                MPI_COMM_WORLD, 
                &requests[i]);
        }
        
        //mpi_barrier();





        u32 recv_loc_cnt = -1;

        printf("n°%d {%d,%d,%d,%d}\n",world_rank,send_loc_cnt[0],send_loc_cnt[1],send_loc_cnt[2],send_loc_cnt[3]);

        mpi_barrier();

        int recv_cnt[] = {1,1,1,1};

        MPI_Reduce_scatter(send_loc_cnt.data(), &recv_loc_cnt, recv_cnt, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

        printf("n°%d : %d\n",world_rank,recv_loc_cnt);


        std::vector<MPI_Packet> payload_recv_table(recv_loc_cnt); 

        for(u32 i = 0; i < recv_loc_cnt;i++){
            //printf("MPI probe rank = %d n°%d\n",world_rank,i);
            MPI_Status st;

            //no race condition beacause this call is blocking
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, & st);
            //printf("MPI probe rank = %d n°%d : result incomming from %d\n",world_rank,i,st.MPI_SOURCE);

            int sz_recv;
            MPI_Get_count(&st, MPI_CHAR, &sz_recv);
            payload_recv_table[i].rank = st.MPI_SOURCE;
            //printf("MPI_Get_count rank = %d n°%d : result incomming from %d : sz : %d\n",world_rank,i,st.MPI_SOURCE,sz_recv);

            MPI_Status st_recv;
            payload_recv_table[i].patchdata = std::vector<u8>(sz_recv);
            MPI_Recv(payload_recv_table[i].patchdata.data(), sz_recv, MPI_CHAR, st.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st_recv);




            
        }

        for(u32 i = 0; i < payload_send_queue.size();i++){
            //printf("MPI wait rank = %d n°%d\n",world_rank,i);
            MPI_Status st;
            MPI_Wait(&requests[i], &st);

            //printf("MPI wait rank = %d n°%d (done)\n",world_rank,i);
        }

        for(u32 i = 0; i < recv_loc_cnt;i++){
            printf("[%d] packet recv : {rank : %d, data : {",world_rank, payload_recv_table[i].rank);
            for(u32 j = 0 ; j < payload_recv_table[i].patchdata.size(); j++){
                printf("%d ",payload_recv_table[i].patchdata[j]);
            }
            printf("}}\n");
        }

    }unit_test::test_end();

}