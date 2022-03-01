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

#include "aliases.hpp"
#include "flags.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include <mpi.h>
#include <vector>
#include <random>

#include "utils/sycl_vector_utils.hpp"

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
    void set(u32 arg_nVarpos_s, u32 arg_nVarpos_d, u32 arg_nVarU1_s, u32 arg_nVarU1_d, u32 arg_nVarU3_s,
                    u32 arg_nVarU3_d);

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
    std::vector<f32>  U1_s  ; ///< f32 's for internal fields  
    std::vector<f64>  U1_d  ; ///< f64 's for internal fields  
    std::vector<f32_3> U3_s ; ///< f32_3 's for internal fields
    std::vector<f64_3> U3_d ; ///< f64_3 's for internal fields
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
 * //TODO find better way to do it : due to the async aspect returning a value is sketchy
 * 
 * @param rq_lst reference to the vector of MPI_Request corresponding to the recv
 * @param rank_source rank to receive from
 * @param tag MPI communication tag
 * @param comm  MPI communicator
 * @return the received patchdata (it works but weird because asynchronous)
 */
PatchData patchdata_irecv( std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm);


/**
 * @brief generate dummy patchdata from a mersen twister
 * 
 * @param eng the mersen twister
 * @return PatchData the generated PatchData
 */
PatchData patchdata_gen_dummy_data(std::mt19937& eng);

/**
 * @brief check if two PatchData content match
 * 
 * @param p1 
 * @param p2 
 * @return true 
 * @return false 
 */
bool patch_data_check_match(PatchData& p1, PatchData& p2);