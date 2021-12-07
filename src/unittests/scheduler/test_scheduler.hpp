
#include "../unit_test_handler.hpp"
#include "../../sys/mpi_handler.hpp"

#include "../../com/sparse_mpi.hpp"

#include <cstdio>
#include <mpi.h>
#include <valarray>
#include <vector>

#include "../../scheduler/scheduler.hpp"

#include <random>

inline Patch create_test_patch_data(u32 i){
    Patch loc_patch;
    loc_patch.id_patch = i*1000 + i*i;
    loc_patch.id_parent = 1;
    loc_patch.id_child_r = 222;
    loc_patch.id_child_l = 111;
    loc_patch.x_min = 0.f;
    loc_patch.y_min = 0.f;
    loc_patch.z_min = 0.f;
    loc_patch.x_max = 0.f;
    loc_patch.y_max = 0.f;
    loc_patch.z_max = 0.f;
    loc_patch.data_count = 7+i*77;

    loc_patch.node_owner_id = i;

    bool should_merge_child = (3*i)%1 == 0;
    bool should_split = (1+2*i)%1 == 0;

    loc_patch.flags = should_merge_child + 2*should_split;

    return loc_patch;
}

inline void run_tests_scheduler(){


    if(unit_test::test_start("tree/mpi_scheduler.hpp::get_patch_count_from_local()", true)){

        scheduler::patch_table_local.resize(world_rank % 3 + 1);
        
        unit_test::test_assert("corect return", scheduler::get_patch_count_from_local() == 
            3*u64((world_size)/3) + 2*u64((world_size+1)/3) + u64((world_size+2)/3));

    }unit_test::test_end();

    //TODO write better test
    if(unit_test::test_start("tree/mpi_scheduler.hpp::rebuild_global_patch_table_from_local()", true)){

        Patch loc_patch = create_test_patch_data(world_rank);
        
        scheduler::patch_table_local.resize(1);
        scheduler::patch_table_local[0] = loc_patch;

        // if(world_rank == 0){
        //     patch_table_local.push_back(create_test_patch_data(100));
        // }

        // if(world_rank == 1){
        //     patch_table_local.push_back(create_test_patch_data(10000));
        // }

        //patch_table.resize(world_size+2);
        scheduler::patch_table.resize(world_size);

        printf("creating patch type\n");
        create_MPI_patch_type();

        printf("exchange patchs\n");
        scheduler::rebuild_global_patch_table_from_local();

        printf("free patch type\n");
        free_MPI_patch_type();

        for(u32 i = 0 ; i < scheduler::patch_table.size(); i++){
            bool is_patch_corect = scheduler::patch_table[i] == create_test_patch_data(i);
            unit_test::test_assert("ALL GATHER corect behavior", is_patch_corect);

            //printf("is patch corect: %d\n",is_patch_corect);

            printf("patch[%d].id = %zu\n" ,i, scheduler::patch_table[i].id_patch);
        }

        scheduler::patch_table.clear();
        scheduler::patch_table_local.clear();

    }unit_test::test_end();

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



    if(unit_test::test_start("tree/mpi_scheduler.hpp::balance_patch_load()", false)){

        u32 world_size_test = 5;

        std::vector<Patch> ptch_tbl;

        std::mt19937 rng(10000);
        std::uniform_int_distribution<int> uni(0,world_size_test-1);
        std::uniform_int_distribution<int> uni_dt_cnt(0,1e7);

        for(u32 i = 0 ; i < 10; i ++){
            Patch p = create_test_patch_data(uni(rng));
            p.data_count = uni_dt_cnt(rng);
            ptch_tbl.push_back(p);
        }

        for(Patch c : ptch_tbl){
            printf(" -> (%d, %u)\n",c.node_owner_id,c.data_count);
        }

        std::vector<Patch>* patch_distrib = new std::vector<Patch>[world_size_test];
        u64* node_load = new u64[world_size_test];

        for(u32 node_id = 0 ; node_id < world_size_test; node_id ++){
            node_load[node_id] = 0;
        }

        for(Patch c : ptch_tbl){
            node_load[c.node_owner_id] += c.data_count;
            patch_distrib[c.node_owner_id].push_back(c);
        }

        for(u32 it = 0 ; it < 10; it ++){

            u64 least_loadded_node_dtcnt = -1;
            u32 least_loadded_node_id = -1;

            u64 most_loadded_node_dtcnt = 0 ;
            u32 most_loadded_node_id = -1;

            for(u32 node_id = 0 ; node_id < world_size_test; node_id ++){
                if(node_load[node_id] < least_loadded_node_dtcnt){
                    least_loadded_node_id = node_id;
                    least_loadded_node_dtcnt = node_load[node_id];
                }

                if(node_load[node_id] > most_loadded_node_dtcnt){
                    most_loadded_node_id = node_id;
                    most_loadded_node_dtcnt = node_load[node_id];
                }
            }
            
            printf("least loaded : %d ,dtcnt = %zu\n",least_loadded_node_id,least_loadded_node_dtcnt);
            printf("most  loaded : %d ,dtcnt = %zu\n",most_loadded_node_id,most_loadded_node_dtcnt);

            u64 delta_dtcnt = most_loadded_node_dtcnt - least_loadded_node_dtcnt;

            printf("delta dtcnt = %zu\n",delta_dtcnt);

            printf("cnt = %zu\n",patch_distrib[most_loadded_node_id].size());

            u64 id_patch_to_move = -1;
            u64 dt_cnt_delta_min = -1;
            for(u32 patch_in_node_id = 0 ; patch_in_node_id < patch_distrib[most_loadded_node_id].size(); patch_in_node_id ++){
                i64 delta_choice_ = std::labs(i64(patch_distrib[most_loadded_node_id][patch_in_node_id].data_count) - i64(delta_dtcnt));
                if(delta_choice_ < dt_cnt_delta_min){
                    id_patch_to_move = patch_in_node_id;
                    dt_cnt_delta_min = delta_choice_;
                }

                printf(" -> (%d, %ld)\n",patch_in_node_id,delta_choice_);
            }

            printf("brest move loaded : %d ,dtcnt = %zu\n",id_patch_to_move,dt_cnt_delta_min);

            //do move

            Patch to_move = patch_distrib[most_loadded_node_id][id_patch_to_move];
            patch_distrib[most_loadded_node_id].erase(patch_distrib[most_loadded_node_id].begin()+id_patch_to_move);

            node_load[most_loadded_node_id] -= to_move.data_count;

            patch_distrib[least_loadded_node_id].push_back(to_move);
            node_load[least_loadded_node_id] += to_move.data_count;

        }
        



        // scheduler::balance_patch_load(std::vector<Patch> &patch_table, u32 world_size)

    }unit_test::test_end();
}