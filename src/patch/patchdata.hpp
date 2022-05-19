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
#include "patch/patchdata_field.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "utils/sycl_vector_utils.hpp"

#include "patchdata_layout.hpp"


/**
 * @brief PatchData container class, the layout is described in patchdata_layout
 */
class PatchData {
  public:
    PatchDataLayout & patchdata_layout;

    std::vector<PatchDataField<f32   >> fields_f32;
    std::vector<PatchDataField<f32_2 >> fields_f32_2;
    std::vector<PatchDataField<f32_3 >> fields_f32_3;
    std::vector<PatchDataField<f32_4 >> fields_f32_4;
    std::vector<PatchDataField<f32_8 >> fields_f32_8;
    std::vector<PatchDataField<f32_16>> fields_f32_16;

    std::vector<PatchDataField<f64   >> fields_f64;
    std::vector<PatchDataField<f64_2 >> fields_f64_2;
    std::vector<PatchDataField<f64_3 >> fields_f64_3;
    std::vector<PatchDataField<f64_4 >> fields_f64_4;
    std::vector<PatchDataField<f64_8 >> fields_f64_8;
    std::vector<PatchDataField<f64_16>> fields_f64_16;

    std::vector<PatchDataField<u32   >> fields_u32;

    std::vector<PatchDataField<u64   >> fields_u64;

    inline PatchData(PatchDataLayout & pdl) : patchdata_layout(pdl){

        for (auto a : pdl.fields_f32) {
            fields_f32.push_back(PatchDataField<f32>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f32_2) {
            fields_f32_2.push_back(PatchDataField<f32_2>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f32_3) {
            fields_f32_3.push_back(PatchDataField<f32_3>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f32_4) {
            fields_f32_4.push_back(PatchDataField<f32_4>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f32_8) {
            fields_f32_8.push_back(PatchDataField<f32_8>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f32_16) {
            fields_f32_16.push_back(PatchDataField<f32_16>(a.name,a.nvar));
        }


        for (auto a : pdl.fields_f64) {
            fields_f64.push_back(PatchDataField<f64>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f64_2) {
            fields_f64_2.push_back(PatchDataField<f64_2>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f64_3) {
            fields_f64_3.push_back(PatchDataField<f64_3>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f64_4) {
            fields_f64_4.push_back(PatchDataField<f64_4>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f64_8) {
            fields_f64_8.push_back(PatchDataField<f64_8>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_f64_16) {
            fields_f64_16.push_back(PatchDataField<f64_16>(a.name,a.nvar));
        }


        for (auto a : pdl.fields_u32) {
            fields_u32.push_back(PatchDataField<u32>(a.name,a.nvar));
        }

        for (auto a : pdl.fields_u64) {
            fields_u64.push_back(PatchDataField<u64>(a.name,a.nvar));
        }
    }

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
PatchData patchdata_gen_dummy_data(PatchDataLayout & pdl, std::mt19937 &eng);

/**
 * @brief check if two PatchData content match
 *
 * @param p1
 * @param p2
 * @return true
 * @return false
 */
bool patch_data_check_match(PatchData &p1, PatchData &p2);

