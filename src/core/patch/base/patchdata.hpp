// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

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

#include <random>
#include <vector>

#include "aliases.hpp"
#include "flags.hpp"
#include "patchdata_field.hpp"
#include "core/sys/mpi_handler.hpp"
#include "core/sys/sycl_mpi_interop.hpp"
#include "core/utils/sycl_vector_utils.hpp"

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
    void extract_element(u32 pidx, PatchData & out_pdat);

    void insert_elements(PatchData & pdat);

    void resize(u32 new_obj_cnt);


    template<class Tvecbox>
    void split_patchdata(PatchData & pd0,PatchData & pd1,PatchData & pd2,PatchData & pd3,PatchData & pd4,PatchData & pd5,PatchData & pd6,PatchData & pd7,
        Tvecbox bmin_p0,Tvecbox bmin_p1,Tvecbox bmin_p2,Tvecbox bmin_p3,Tvecbox bmin_p4,Tvecbox bmin_p5,Tvecbox bmin_p6,Tvecbox bmin_p7,
        Tvecbox bmax_p0,Tvecbox bmax_p1,Tvecbox bmax_p2,Tvecbox bmax_p3,Tvecbox bmax_p4,Tvecbox bmax_p5,Tvecbox bmax_p6,Tvecbox bmax_p7);
    
    void append_subset_to(std::vector<u32> & idxs, PatchData & pdat);

    inline u32 get_obj_cnt(){
        u32 ret;
        if(patchdata_layout.xyz_mode == xyz32){
            u32 ixyz = patchdata_layout.get_field_idx<f32_3>("xyz");
            ret = fields_f32_3[ixyz].get_obj_cnt();
        }else if(patchdata_layout.xyz_mode == xyz64){
            u32 ixyz = patchdata_layout.get_field_idx<f64_3>("xyz");
            ret = fields_f64_3[ixyz].get_obj_cnt();
        }
        return ret;
    }

    inline bool is_empty(){
        return get_obj_cnt() == 0;
    }

    /*
    inline void expand(u32 obj_to_add){
        for (auto a : fields_f32) {
            a.expand(obj_to_add);
        }

        for (auto a : fields_f32_2) {
            a.expand(obj_to_add);
        }

        for (auto a : fields_f32_3) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_f32_4) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_f32_8) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_f32_16) {
            a.expand(obj_to_add);        
        }


        for (auto a : fields_f64) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_f64_2) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_f64_3) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_f64_4) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_f64_8) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_f64_16) {
            a.expand(obj_to_add);        
        }


        for (auto a : fields_u32) {
            a.expand(obj_to_add);        
        }

        for (auto a : fields_u64) {
            a.expand(obj_to_add);        
        }
    }
    */











    template<class T> PatchDataField<T> & get_field(u32 idx){}

    template<> inline PatchDataField<f32   > & get_field(u32 idx){return fields_f32.at(idx);}
    template<> inline PatchDataField<f32_2 >& get_field(u32 idx){return fields_f32_2.at(idx);}
    template<> inline PatchDataField<f32_3 >& get_field(u32 idx){return fields_f32_3.at(idx);}
    template<> inline PatchDataField<f32_4 >& get_field(u32 idx){return fields_f32_4.at(idx);}
    template<> inline PatchDataField<f32_8 >& get_field(u32 idx){return fields_f32_8.at(idx);}
    template<> inline PatchDataField<f32_16>& get_field(u32 idx){return fields_f32_16.at(idx);}
    template<> inline PatchDataField<f64   >& get_field(u32 idx){return fields_f64.at(idx);}
    template<> inline PatchDataField<f64_2 >& get_field(u32 idx){return fields_f64_2.at(idx);}
    template<> inline PatchDataField<f64_3 >& get_field(u32 idx){return fields_f64_3.at(idx);}
    template<> inline PatchDataField<f64_4 >& get_field(u32 idx){return fields_f64_4.at(idx);}
    template<> inline PatchDataField<f64_8 >& get_field(u32 idx){return fields_f64_8.at(idx);}
    template<> inline PatchDataField<f64_16>& get_field(u32 idx){return fields_f64_16.at(idx);}
    template<> inline PatchDataField<u32   >& get_field(u32 idx){return fields_u32.at(idx);}
    template<> inline PatchDataField<u64   >& get_field(u32 idx){return fields_u64.at(idx);}
    
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
u64 patchdata_isend(PatchData &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm);

/**
 * @brief perform a MPI irecv with a PatchData object
 *
 * @param rq_lst reference to the vector of MPI_Request corresponding to the recv
 * @param rank_source rank to receive from
 * @param tag MPI communication tag
 * @param comm  MPI communicator
 * @return the received patchdata (it works but weird because asynchronous)
 */
u64 patchdata_irecv(PatchData &pdat, std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm);

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

