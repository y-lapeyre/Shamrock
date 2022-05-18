/**
 * @file patchdata.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief header for PatchData related function and declaration
 * @version 0.1
 * @date 2022-02-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <mpi.h>
#include <random>
#include <vector>

#include "CL/sycl/usm.hpp"
#include "aliases.hpp"
#include "flags.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "utils/sycl_vector_utils.hpp"


//TODO study if patchdata can be templated by the patchdata layout to unroll for loop on nvar

/**
 * @brief manage the information on the layout of patchdata
 */
namespace patchdata_layout {

inline u32 nVarpos_s; ///< number of f32 per object for position
inline u32 nVarpos_d; ///< number of f64 per object for position
inline u32 nVarU1_s;  ///< number of f32 per object for internal fields
inline u32 nVarU1_d;  ///< number of f64 per object for internal fields
inline u32 nVarU3_s;  ///< number of f32_3 per object for internal fields
inline u32 nVarU3_d;  ///< number of f64_3 per object for internal fields

/**
 * @brief should be check if true before communication with patchdata_s
 */
inline bool layout_synced = false;

/**
 * @brief sync the patchdata layout accors the MPI communicator \p comm
 *
 * @param comm the MPI communicator
 */
void sync(MPI_Comm comm);

/**
 * @brief set the patchdata layout on this node
 *
 * @param arg_nVarpos_s ///< number of f32 per object for position
 * @param arg_nVarpos_d ///< number of f64 per object for position
 * @param arg_nVarU1_s  ///< number of f32 per object for internal fields
 * @param arg_nVarU1_d  ///< number of f64 per object for internal fields
 * @param arg_nVarU3_s  ///< number of f32_3 per object for internal fields
 * @param arg_nVarU3_d  ///< number of f64_3 per object for internal fields
 */
void set(u32 arg_nVarpos_s, u32 arg_nVarpos_d, u32 arg_nVarU1_s, u32 arg_nVarU1_d, u32 arg_nVarU3_s, u32 arg_nVarU3_d);

/**
 * @brief should be check before using the layout
 *
 * //TODO add runtime exception check to function using it
 *
 * @return true patchdata_layout is synced
 * @return false  patchdata_layout isnt synced
 */
bool is_synced();

} // namespace patchdata_layout

/**
 * @brief PatchData container class, the layout is described in patchdata_layout
 */
class PatchData {
  public:
    std::vector<f32_3> pos_s; ///< f32 's for position
    std::vector<f64_3> pos_d; ///< f64 's for position
    std::vector<f32> U1_s;    ///< f32 's for internal fields
    std::vector<f64> U1_d;    ///< f64 's for internal fields
    std::vector<f32_3> U3_s;  ///< f32_3 's for internal fields
    std::vector<f64_3> U3_d;  ///< f64_3 's for internal fields

    /**
     * @brief extract particle at index pidx and insert it in the provided vectors
     * 
     * @param pidx 
     * @param out_pos_s 
     * @param out_pos_d 
     * @param out_U1_s 
     * @param out_U1_d 
     * @param out_U3_s 
     * @param out_U3_d 
     */
    void extract_particle(u32 pidx, std::vector<f32_3> &out_pos_s, std::vector<f64_3> &out_pos_d, std::vector<f32> &out_U1_s,
                          std::vector<f64> &out_U1_d, std::vector<f32_3> &out_U3_s, std::vector<f64_3> &out_U3_d);

    void insert_particles(std::vector<f32_3> &in_pos_s, std::vector<f64_3> &in_pos_d, std::vector<f32> &in_U1_s,
                          std::vector<f64> &in_U1_d, std::vector<f32_3> &in_U3_s, std::vector<f64_3> &in_U3_d);
};

/**
 * @brief perform a MPI isend with a PatchData object
 *
 * @param p the patchdata to send
 * @param rq_lst reference to the vector of MPI_Request corresponding to the send
 * @param rank_dest rabk to send data to
 * @param tag MPI communication tag
 * @param comm MPI communicator
 */
void patchdata_isend(PatchData &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm);

/**
 * @brief perform a MPI irecv with a PatchData object
 *
 * @param rq_lst reference to the vector of MPI_Request corresponding to the recv
 * @param rank_source rank to receive from
 * @param tag MPI communication tag
 * @param comm  MPI communicator
 * @return the received patchdata (it works but weird because asynchronous)
 */
void patchdata_irecv(PatchData &pdat, std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm);

/**
 * @brief generate dummy patchdata from a mersen twister
 *
 * @param eng the mersen twister
 * @return PatchData the generated PatchData
 */
PatchData patchdata_gen_dummy_data(std::mt19937 &eng);

/**
 * @brief check if two PatchData content match
 *
 * @param p1
 * @param p2
 * @return true
 * @return false
 */
bool patch_data_check_match(PatchData &p1, PatchData &p2);


// TODO Make & Test of the new patchdata model

enum FieldAllocMode{
    SharedUSM,
    DeviceUSM,
    DeviceMPIUSM
};

template<class T, FieldAllocMode alloctype>
class Field{public:

    std::vector<std::string> name_list;

    u32 nvar;
    u32 obj_cnt;

    u32 current_lenght;
    u32 storage_capacity;

    u64 memsize;

    T* field_data = nullptr;


    sycl::queue & owner_queue;


    inline T* _alloc_buf(u32 sz){
        T* ret;
        if constexpr (alloctype == SharedUSM){
            ret = sycl::malloc_shared<T>(sz, owner_queue);
        }else if constexpr (alloctype == DeviceUSM) {
            ret = sycl::malloc_device<T>(sz, owner_queue);
        }else if constexpr (alloctype == DeviceMPIUSM) {
            ret = sycl::malloc_device<T>(sz, owner_queue);
        }
        return ret;
    }

    inline void _free_buf(){
        sycl::free(field_data, owner_queue);
    }




    inline Field(){

    }

    inline T* data(){
        return field_data;
    }

    inline void resize(u32 obj_count){

        u32 new_len = obj_cnt*nvar;

        if(new_len > storage_capacity){
            T* new_buf = _alloc_buf(new_len);
            //TODO finish
        }



        obj_cnt = obj_count;
        current_lenght = obj_cnt*nvar;
    }









    T* host_storage_buf = nullptr;

    inline void mpi_send(int dest, int tag, MPI_Comm comm){
        if constexpr (alloctype == SharedUSM){
            mpi::send(field_data, current_lenght,get_mpi_type<T>(), dest, tag, comm);
        }else if constexpr (alloctype == DeviceUSM) {

            host_storage_buf = new T[current_lenght];

            owner_queue.memcpy(host_storage_buf, field_data, current_lenght*sizeof(T));

            owner_queue.wait();

            mpi::send(host_storage_buf, current_lenght,get_mpi_type<T>(), dest, tag, comm);

            delete[] host_storage_buf;

        }else if constexpr (alloctype == DeviceMPIUSM) {
            mpi::send(field_data, current_lenght,get_mpi_type<T>(), dest, tag, comm);
        }
    }


    inline MPI_Status mpi_recv(int source, int tag, MPI_Comm comm){
        MPI_Status st;
        if constexpr (alloctype == SharedUSM){
            mpi::recv(field_data, current_lenght,get_mpi_type<T>(), source, tag, comm, &st);
        }else if constexpr (alloctype == DeviceUSM) {

            host_storage_buf = new T[current_lenght];

            mpi::recv(host_storage_buf, current_lenght,get_mpi_type<T>(), source, tag, comm, &st);

            owner_queue.memcpy(field_data,host_storage_buf, current_lenght*sizeof(T));

            owner_queue.wait();

            delete[] host_storage_buf;

        }else if constexpr (alloctype == DeviceMPIUSM) {
            mpi::recv(field_data, current_lenght,get_mpi_type<T>(), source, tag, comm, &st);
        }
    }








};

// To be renmed PatchData in the end
class PatchDataUSM{

};