/**
 * @file mpi_handler.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief handle mpi routines
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once



#include <vector>
//#define MPI_LOGGER_ENABLED

#include <cstdio>
#include "mpi_wrapper.hpp"
#include "../utils/string_utils.hpp"

#include "../aliases.hpp"

namespace mpi_handler{

    inline bool working = false;

    inline i32 world_rank, world_size;
    //inline Logger* global_logger;

    /**
    * @brief Get the processor name 
    * @return std::string processor name
    */
    std::string get_proc_name();

    /**
    * @brief initialize MPI comm and logger
    */
    void init();

    /**
    * @brief close MPI context
    * 
    */
    void close();





    



    /**
     * @brief allgatherv with knowing total count of object
     * //TODO add fault tolerance
     * @tparam T 
     * @param send_vec 
     * @param send_type 
     * @param recv_vec 
     * @param recv_type 
     */
    template<class T>
    inline void vector_allgatherv_ks(const std::vector<T> & send_vec ,const MPI_Datatype send_type,std::vector<T> & recv_vec,const MPI_Datatype recv_type,const MPI_Comm comm){

        u32 local_count = send_vec.size();

        int* table_data_count = new int[world_size];

        //crash
        mpi::allgather(
            &local_count, 
            1, 
            MPI_INT, 
            &table_data_count[0], 
            1, 
            MPI_INT, 
            comm);

        //printf("table_data_count = [%d,%d,%d,%d]\n",table_data_count[0],table_data_count[1],table_data_count[2],table_data_count[3]);



        int* node_displacments_data_table = new int[world_size];

        node_displacments_data_table[0] = 0;

        for(i32 i = 1 ; i < world_size; i++){
            node_displacments_data_table[i] = node_displacments_data_table[i-1] + table_data_count[i-1];
        }
        
        //printf("node_displacments_data_table = [%d,%d,%d,%d]\n",node_displacments_data_table[0],node_displacments_data_table[1],node_displacments_data_table[2],node_displacments_data_table[3]);
        
        mpi::allgatherv(
            &send_vec[0], 
            send_vec.size(),
            send_type, 
            &recv_vec[0], 
            table_data_count, 
            node_displacments_data_table, 
            recv_type, 
            comm);


        delete [] table_data_count;
        delete [] node_displacments_data_table;
    }





    /**
     * @brief allgatherv on vector with size query (size querrying variant of vector_allgatherv_ks)
     * //TODO add fault tolerance
     * @tparam T 
     * @param send_vec 
     * @param send_type 
     * @param recv_vec 
     * @param recv_type 
     */
    template<class T>
    inline void vector_allgatherv(const std::vector<T> & send_vec ,const MPI_Datatype & send_type,std::vector<T> & recv_vec,const MPI_Datatype & recv_type,const MPI_Comm comm){

        u32 local_count = send_vec.size();


        //querry global size and resize the receiving vector
        u32 global_len;
        mpi::allreduce(&local_count, &global_len, 1, MPI_INT , MPI_SUM, MPI_COMM_WORLD);
        recv_vec.resize(global_len);



        int* table_data_count = new int[world_size];

        mpi::allgather(
            &local_count, 
            1, 
            MPI_INT, 
            &table_data_count[0], 
            1, 
            MPI_INT, 
            comm);

        //printf("table_data_count = [%d,%d,%d,%d]\n",table_data_count[0],table_data_count[1],table_data_count[2],table_data_count[3]);



        int* node_displacments_data_table = new int[world_size];

        node_displacments_data_table[0] = 0;

        for(i32 i = 1 ; i < world_size; i++){
            node_displacments_data_table[i] = node_displacments_data_table[i-1] + table_data_count[i-1];
        }
        
        //printf("node_displacments_data_table = [%d,%d,%d,%d]\n",node_displacments_data_table[0],node_displacments_data_table[1],node_displacments_data_table[2],node_displacments_data_table[3]);
        
        mpi::allgatherv(
            &send_vec[0], 
            send_vec.size(),
            send_type, 
            &recv_vec[0], 
            table_data_count, 
            node_displacments_data_table, 
            recv_type, 
            comm);


        delete [] table_data_count;
        delete [] node_displacments_data_table;
    } 




    /**
     * @brief perform a alltoall with varying send and receive per node using a sparse methode
     * //TODO add fault tolerance
     * @tparam T 
     * @param send_arr_node_id 
     * @param send_arr_tag 
     * @param send_arr_data 
     * @param exchange_datatype MPI_Datatype to use
     * @param recv_arr_node_id 
     * @param recv_arr_tag 
     * @param recv_arr_data 
     * @param node_cnt Node count on the MPI communicator
     * @param comm MPI communicator to use
     */
    template<class T>
    inline void sparse_alltoall(
        const std::vector<       u32       > & send_arr_node_id,
        const std::vector<       u32       > & send_arr_tag,
        const std::vector<  std::vector<T> > & send_arr_data,
        
        const MPI_Datatype exchange_datatype,

        std::vector<       u32       > & recv_arr_node_id,
        std::vector<       u32       > & recv_arr_tag,
        std::vector<  std::vector<T> > & recv_arr_data,

        const u32 node_cnt,
        const MPI_Comm comm
        ){


        std::vector<u32> send_loc_cnt(node_cnt,0);
        std::vector<MPI_Request> requests_send(send_arr_node_id.size());
        
        for(u32 i = 0; i < send_arr_node_id.size();i++){
            //printf("async send rank = %d n°%d\n",mpi::world_rank,i);
            mpi::isend(
                send_arr_data[i].data(), 
                send_arr_data[i].size(), 
                exchange_datatype, 
                send_arr_node_id[i], 
                send_arr_tag[i], 
                comm, 
                &requests_send[i]);

            send_loc_cnt[send_arr_node_id[i]] ++;
        }

        //wait for the end of ISend calls 
        for(u32 i = 0; i < requests_send.size();i++){
            MPI_Status st;
            mpi::wait(&requests_send[i], &st);
        }
        

        u32 recv_loc_cnt = -1;
        {
            const std::vector<int> recv_cnt(node_cnt,1);
            mpi::reduce_scatter(send_loc_cnt.data(), &recv_loc_cnt, recv_cnt.data(), MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
        }

        printf("will receive : %d\n",recv_loc_cnt);


        recv_arr_node_id.resize(recv_loc_cnt);
        recv_arr_tag.resize(recv_loc_cnt);
        recv_arr_data.resize(recv_loc_cnt);







        std::vector<MPI_Request> requests_recv(recv_loc_cnt);


        /*asynchronous probe seems unapropriate
        * sticking with synchronous probe for now 
        */
        for(u32 i = 0; i < recv_loc_cnt;i++){
            MPI_Status st;

            //no race condition beacause this call is blocking
            mpi::probe(MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, & st);

            recv_arr_node_id[i] = st.MPI_SOURCE;
            recv_arr_tag[i]     = st.MPI_TAG;

            int sz_recv;
            mpi::get_count(&st, exchange_datatype, &sz_recv);

            recv_arr_data[i].resize(sz_recv);
            
            //MPI_Status st_recv;
            //mpi::recv(recv_arr_data[i].data(), sz_recv, exchange_datatype, st.MPI_SOURCE, st.MPI_TAG, comm, &st_recv);
            mpi::irecv(recv_arr_data[i].data(), sz_recv, exchange_datatype, st.MPI_SOURCE, st.MPI_TAG, comm, &requests_recv[i]);
        }


        //wait for the end of IRecv calls 
        //*
        for(u32 i = 0; i < requests_recv.size();i++){
            MPI_Status st;
            mpi::wait(&requests_recv[i], &st);
        }
        
        //*/

        

    }





}
