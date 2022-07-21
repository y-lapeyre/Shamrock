// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patchdata.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief implementation of PatchData related functions
 * @version 0.1
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "patchdata.hpp"
#include "aliases.hpp"
#include "patchdata_field.hpp"
#include "core/sys/mpi_handler.hpp"
#include "core/sys/sycl_mpi_interop.hpp"
#include "core/utils/geometry_utils.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>



void PatchData::init_fields(){

    #define X(arg) \
        for (auto a : pdl.fields_##arg) {\
            fields_##arg.emplace_back(a.name,a.nvar);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X

}



u64 patchdata_isend(PatchData &p, std::vector<PatchDataMpiRequest> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm) {

    rq_lst.resize(rq_lst.size()+1);
    auto & ref = rq_lst[rq_lst.size()-1];

    u64 total_data_transf = 0;

    #define X(arg) \
        for (auto & a : p.fields_##arg) {\
            total_data_transf += patchdata_field::isend(a,ref.mpi_rq_fields_##arg, rank_dest, tag, comm);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X


    return total_data_transf;
}




u64 patchdata_irecv(PatchData & pdat, std::vector<PatchDataMpiRequest> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){

    rq_lst.resize(rq_lst.size()+1);
    auto & ref = rq_lst[rq_lst.size()-1];

    u64 total_data_transf = 0;

    #define X(arg) \
        for (auto & a : pdat.fields_##arg) {\
            total_data_transf += patchdata_field::irecv(a, ref.mpi_rq_fields_##arg, rank_source, tag, comm);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X

    return total_data_transf;

}


PatchData patchdata_gen_dummy_data(PatchDataLayout & pdl, std::mt19937& eng){

    std::uniform_int_distribution<u64> distu64(1,6000);

    u32 num_part = distu64(eng);

    PatchData pdat(pdl);


    #define X(arg) \
        for (auto & a : pdat.fields_##arg) {\
            a.gen_mock_data(num_part, eng);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X


    return pdat;
}


bool patch_data_check_match(PatchData& p1, PatchData& p2){
    bool check = true;

    #define X(arg) \
        for(u32 idx = 0; idx < p1.pdl.fields_##arg.size(); idx++){\
            check = p1.fields_##arg[idx].check_field_match(p2.fields_##arg[idx]);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X

    return check;
}



void PatchData::extract_element(u32 pidx, PatchData & out_pdat){

    #define X(arg) \
        for(u32 idx = 0; idx < pdl.fields_##arg.size(); idx++){\
            fields_##arg[idx].extract_element(pidx, out_pdat.fields_##arg[idx]);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X

}

void PatchData::insert_elements(PatchData & pdat){

    #define X(arg) \
        for(u32 idx = 0; idx < pdl.fields_##arg.size(); idx++){\
            fields_##arg[idx].insert(pdat.fields_##arg[idx]);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X

}

void PatchData::overwrite(PatchData &pdat, u32 obj_cnt){
    #define X(arg) \
        for(u32 idx = 0; idx < pdl.fields_##arg.size(); idx++){\
            fields_##arg[idx].overwrite(pdat.fields_##arg[idx],obj_cnt);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X
}



void PatchData::resize(u32 new_obj_cnt){

    #define X(arg) \
        for(u32 idx = 0; idx < pdl.fields_##arg.size(); idx++){\
            fields_##arg[idx].resize(new_obj_cnt);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X

}


void PatchData::append_subset_to(std::vector<u32> & idxs, PatchData &pdat){

    #define X(arg) \
        for(u32 idx = 0; idx < pdl.fields_##arg.size(); idx++){\
            fields_##arg[idx].append_subset_to(idxs, pdat.fields_##arg[idx]);\
        }
    XMAC_LIST_ENABLED_FIELD
    #undef X

}

template<>
void PatchData::split_patchdata<f32_3>(
    PatchData &pd0, PatchData &pd1, PatchData &pd2, PatchData &pd3, PatchData &pd4, PatchData &pd5, PatchData &pd6, PatchData &pd7, 
    f32_3 bmin_p0, f32_3 bmin_p1, f32_3 bmin_p2, f32_3 bmin_p3, f32_3 bmin_p4, f32_3 bmin_p5, f32_3 bmin_p6, f32_3 bmin_p7, 
    f32_3 bmax_p0, f32_3 bmax_p1, f32_3 bmax_p2, f32_3 bmax_p3, f32_3 bmax_p4, f32_3 bmax_p5, f32_3 bmax_p6, f32_3 bmax_p7){

    u32 field_ipos = pdl.get_field_idx<f32_3>("xyz");
    //TODO check that nvar on this field is 1 on creation

    auto get_vec_idx = [&](f32_3 vmin, f32_3 vmax) -> std::vector<u32>{
        return fields_f32_3[field_ipos].get_elements_with_range(
            [&](f32_3 val,f32_3 vmin, f32_3 vmax){
                return BBAA::is_particle_in_patch<f32_3>(val, vmin,vmax);
            },
            vmin,vmax
        );
    };

    std::vector<u32> idx_p0 = get_vec_idx(bmin_p0,bmax_p0);
    std::vector<u32> idx_p1 = get_vec_idx(bmin_p1,bmax_p1);
    std::vector<u32> idx_p2 = get_vec_idx(bmin_p2,bmax_p2);
    std::vector<u32> idx_p3 = get_vec_idx(bmin_p3,bmax_p3);
    std::vector<u32> idx_p4 = get_vec_idx(bmin_p4,bmax_p4);
    std::vector<u32> idx_p5 = get_vec_idx(bmin_p5,bmax_p5);
    std::vector<u32> idx_p6 = get_vec_idx(bmin_p6,bmax_p6);
    std::vector<u32> idx_p7 = get_vec_idx(bmin_p7,bmax_p7);

    //TODO create a extract subpatch function

    append_subset_to(idx_p0, pd0);
    append_subset_to(idx_p1, pd1);
    append_subset_to(idx_p2, pd2);
    append_subset_to(idx_p3, pd3);
    append_subset_to(idx_p4, pd4);
    append_subset_to(idx_p5, pd5);
    append_subset_to(idx_p6, pd6);
    append_subset_to(idx_p7, pd7);

}


template<>
void PatchData::split_patchdata<f64_3>(
    PatchData &pd0, PatchData &pd1, PatchData &pd2, PatchData &pd3, PatchData &pd4, PatchData &pd5, PatchData &pd6, PatchData &pd7, 
    f64_3 bmin_p0, f64_3 bmin_p1, f64_3 bmin_p2, f64_3 bmin_p3, f64_3 bmin_p4, f64_3 bmin_p5, f64_3 bmin_p6, f64_3 bmin_p7, 
    f64_3 bmax_p0, f64_3 bmax_p1, f64_3 bmax_p2, f64_3 bmax_p3, f64_3 bmax_p4, f64_3 bmax_p5, f64_3 bmax_p6, f64_3 bmax_p7){

    u32 field_ipos = pdl.get_field_idx<f64_3>("xyz");
    //TODO check that nvar on this field is 1 on creation

    auto get_vec_idx = [&](f64_3 vmin, f64_3 vmax) -> std::vector<u32>{
        return fields_f64_3[field_ipos].get_elements_with_range(
            [&](f64_3 val,f64_3 vmin, f64_3 vmax){
                return BBAA::is_particle_in_patch<f64_3>(val, vmin,vmax);
            },
            vmin,vmax
        );
    };

    std::vector<u32> idx_p0 = get_vec_idx(bmin_p0,bmax_p0);
    std::vector<u32> idx_p1 = get_vec_idx(bmin_p1,bmax_p1);
    std::vector<u32> idx_p2 = get_vec_idx(bmin_p2,bmax_p2);
    std::vector<u32> idx_p3 = get_vec_idx(bmin_p3,bmax_p3);
    std::vector<u32> idx_p4 = get_vec_idx(bmin_p4,bmax_p4);
    std::vector<u32> idx_p5 = get_vec_idx(bmin_p5,bmax_p5);
    std::vector<u32> idx_p6 = get_vec_idx(bmin_p6,bmax_p6);
    std::vector<u32> idx_p7 = get_vec_idx(bmin_p7,bmax_p7);

    //TODO create a extract subpatch function

    append_subset_to(idx_p0, pd0);
    append_subset_to(idx_p1, pd1);
    append_subset_to(idx_p2, pd2);
    append_subset_to(idx_p3, pd3);
    append_subset_to(idx_p4, pd4);
    append_subset_to(idx_p5, pd5);
    append_subset_to(idx_p6, pd6);
    append_subset_to(idx_p7, pd7);

}