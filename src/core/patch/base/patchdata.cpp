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







u64 patchdata_isend(PatchData &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm) {

    u64 total_data_transf = 0;

    for (auto & a : p.fields_f32) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f32_2) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f32_3) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f32_4) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f32_8) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f32_16) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }


    for (auto & a : p.fields_f64) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f64_2) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f64_3) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f64_4) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f64_8) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_f64_16) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_u32) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    for (auto & a : p.fields_u64) {        //std::cout << "["<< mpi_handler::world_rank <<"] sending field : " << a.get_name() << std::endl;
         total_data_transf += patchdata_field::isend(a,rq_lst, rank_dest, tag, comm);
    }

    return total_data_transf;
}

/*
void patchdata_irecv(PatchData & pdat, std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){

    for (auto & a : pdat.fields_f32) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    for (auto & a : pdat.fields_f32_2) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_2, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_2, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f32_3) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_3, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_3, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f32_4) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_4, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_4, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f32_8) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_8, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_8, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f32_16) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_16, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_16, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }





    for (auto & a : pdat.fields_f64) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    for (auto & a : pdat.fields_f64_2) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_2, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_2, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f64_3) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_3, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_3, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f64_4) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_4, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_4, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f64_8) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_8, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_8, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f64_16) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_16, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_16, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }
    



    for (auto & a : pdat.fields_u32) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_u32, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_u32, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_u64) {        std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_u64, &cnt);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_u64, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


}
*/


u64 patchdata_irecv(PatchData & pdat, std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){

    u64 total_data_transf = 0;

    for (auto & a : pdat.fields_f32) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }


    for (auto & a : pdat.fields_f32_2) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_f32_3) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_f32_4) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_f32_8) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_f32_16) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }





    for (auto & a : pdat.fields_f64) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }


    for (auto & a : pdat.fields_f64_2) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_f64_3) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_f64_4) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_f64_8) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_f64_16) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }
    



    for (auto & a : pdat.fields_u32) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    for (auto & a : pdat.fields_u64) {        //std::cout << "["<< mpi_handler::world_rank <<"] recv field : " << a.get_name() << std::endl;
        total_data_transf += patchdata_field::irecv(a, rq_lst, rank_source, tag, comm);
    }

    return total_data_transf;

}


PatchData patchdata_gen_dummy_data(PatchDataLayout & pdl, std::mt19937& eng){

    std::uniform_int_distribution<u64> distu64(1,6000);

    u32 num_part = distu64(eng);

    PatchData pdat(pdl);


    for (auto & a : pdat.fields_f32) {
        a.gen_mock_data(num_part, eng);
    }

    
    for (auto & a : pdat.fields_f32_2) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f32_3) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f32_4) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f32_8) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f32_16) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f64) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f64_2) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f64_3) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f64_4) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f64_8) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_f64_16) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_u32) {
        a.gen_mock_data(num_part, eng);
    }

    for (auto & a : pdat.fields_u64) {
        a.gen_mock_data(num_part, eng);
    }
    




    return pdat;
}


bool patch_data_check_match(PatchData& p1, PatchData& p2){
    bool check = true;

    for(u32 idx = 0; idx < p1.pdl.fields_f32.size(); idx++){
        check = p1.fields_f32[idx].check_field_match(p2.fields_f32[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f32_2.size(); idx++){
        check = p1.fields_f32_2[idx].check_field_match(p2.fields_f32_2[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f32_3.size(); idx++){
        check = p1.fields_f32_3[idx].check_field_match(p2.fields_f32_3[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f32_4.size(); idx++){
        check = p1.fields_f32_4[idx].check_field_match(p2.fields_f32_4[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f32_8.size(); idx++){
        check = p1.fields_f32_8[idx].check_field_match(p2.fields_f32_8[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f32_16.size(); idx++){
        check = p1.fields_f32_16[idx].check_field_match(p2.fields_f32_16[idx]);
    }



    for(u32 idx = 0; idx < p1.pdl.fields_f64.size(); idx++){
        check = p1.fields_f64[idx].check_field_match(p2.fields_f64[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f64_2.size(); idx++){
        check = p1.fields_f64_2[idx].check_field_match(p2.fields_f64_2[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f64_3.size(); idx++){
        check = p1.fields_f64_3[idx].check_field_match(p2.fields_f64_3[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f64_4.size(); idx++){
        check = p1.fields_f64_4[idx].check_field_match(p2.fields_f64_4[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f64_8.size(); idx++){
        check = p1.fields_f64_8[idx].check_field_match(p2.fields_f64_8[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_f64_16.size(); idx++){
        check = p1.fields_f64_16[idx].check_field_match(p2.fields_f64_16[idx]);
    }



    for(u32 idx = 0; idx < p1.pdl.fields_u32.size(); idx++){
        check = p1.fields_u32[idx].check_field_match(p2.fields_u32[idx]);
    }

    for(u32 idx = 0; idx < p1.pdl.fields_u64.size(); idx++){
        check = p1.fields_u64[idx].check_field_match(p2.fields_u64[idx]);
    }

    return check;
}

template<class obj> obj fast_extract(u32 idx, std::vector<obj>& cnt){

    obj end_ = *(cnt.end()-1);
    obj extr = cnt[idx];

    cnt[idx] = end_;
    cnt.pop_back();

    return extr;
}


template<class obj> obj fast_extract_ptr(u32 idx, u32 lenght ,obj* cnt){

    obj end_ = cnt[lenght-1];
    obj extr = cnt[idx];

    cnt[idx] = end_;

    return extr;
}

void PatchData::extract_element(u32 pidx, PatchData & out_pdat){

    
    for(u32 idx = 0; idx < pdl.fields_f32.size(); idx++){
        const u32 nvar = fields_f32[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f32[idx].size();

        out_pdat.fields_f32[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f32[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f32[idx].size(), fields_f32[idx].usm_data()));
        }

        fields_f32[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_2.size(); idx++){
        const u32 nvar = fields_f32_2[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f32_2[idx].size();

        out_pdat.fields_f32_2[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f32_2[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f32_2[idx].size(), fields_f32_2[idx].usm_data()));
        }

        fields_f32_2[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_3.size(); idx++){
        const u32 nvar = fields_f32_3[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f32_3[idx].size();

        out_pdat.fields_f32_3[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f32_3[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f32_3[idx].size(), fields_f32_3[idx].usm_data()));
        }

        fields_f32_3[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_4.size(); idx++){
        const u32 nvar = fields_f32_4[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f32_4[idx].size();

        out_pdat.fields_f32_4[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f32_4[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f32_4[idx].size(), fields_f32_4[idx].usm_data()));
        }

        fields_f32_4[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_8.size(); idx++){
        const u32 nvar = fields_f32_8[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f32_8[idx].size();

        out_pdat.fields_f32_8[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f32_8[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f32_8[idx].size(), fields_f32_8[idx].usm_data()));
        }

        fields_f32_8[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_16.size(); idx++){
        const u32 nvar = fields_f32_16[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f32_16[idx].size();

        out_pdat.fields_f32_16[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f32_16[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f32_16[idx].size(), fields_f32_16[idx].usm_data()));
        }

        fields_f32_16[idx].shrink(1);
    }






    for(u32 idx = 0; idx < pdl.fields_f64.size(); idx++){
        const u32 nvar = fields_f64[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f64[idx].size();

        out_pdat.fields_f64[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f64[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f64[idx].size(), fields_f64[idx].usm_data()));
        }

        fields_f64[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_2.size(); idx++){
        const u32 nvar = fields_f64_2[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f64_2[idx].size();

        out_pdat.fields_f64_2[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f64_2[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f64_2[idx].size(), fields_f64_2[idx].usm_data()));
        }

        fields_f64_2[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_3.size(); idx++){
        const u32 nvar = fields_f64_3[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f64_3[idx].size();

        out_pdat.fields_f64_3[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f64_3[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f64_3[idx].size(), fields_f64_3[idx].usm_data()));
        }

        fields_f64_3[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_4.size(); idx++){
        const u32 nvar = fields_f64_4[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f64_4[idx].size();

        out_pdat.fields_f64_4[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f64_4[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f64_4[idx].size(), fields_f64_4[idx].usm_data()));
        }

        fields_f64_4[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_8.size(); idx++){
        const u32 nvar = fields_f64_8[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f64_8[idx].size();

        out_pdat.fields_f64_8[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f64_8[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f64_8[idx].size(), fields_f64_8[idx].usm_data()));
        }

        fields_f64_8[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_16.size(); idx++){
        const u32 nvar = fields_f64_16[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_f64_16[idx].size();

        out_pdat.fields_f64_16[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_f64_16[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_f64_16[idx].size(), fields_f64_16[idx].usm_data()));
        }

        fields_f64_16[idx].shrink(1);
    }


    for(u32 idx = 0; idx < pdl.fields_u32.size(); idx++){
        const u32 nvar = fields_u32[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_u32[idx].size();

        out_pdat.fields_u32[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_u32[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_u32[idx].size(), fields_u32[idx].usm_data()));
        }

        fields_u32[idx].shrink(1);
    }

    for(u32 idx = 0; idx < pdl.fields_u64.size(); idx++){
        const u32 nvar = fields_u64[idx].get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = out_pdat.fields_u64[idx].size();

        out_pdat.fields_u64[idx].expand(1);

        for(u32 i = nvar-1 ; i < nvar ; i--){
            out_pdat.fields_u64[idx].usm_data()[idx_out_val + i] = (fast_extract_ptr(idx_val + i,fields_u64[idx].size(), fields_u64[idx].usm_data()));
        }

        fields_u64[idx].shrink(1);
    }



}

void PatchData::insert_elements(PatchData & pdat){
    for(u32 idx = 0; idx < pdl.fields_f32.size(); idx++){
        fields_f32[idx].insert(pdat.fields_f32[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_2.size(); idx++){
        fields_f32_2[idx].insert(pdat.fields_f32_2[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_3.size(); idx++){
        fields_f32_3[idx].insert(pdat.fields_f32_3[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_4.size(); idx++){
        fields_f32_4[idx].insert(pdat.fields_f32_4[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_8.size(); idx++){
        fields_f32_8[idx].insert(pdat.fields_f32_8[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_16.size(); idx++){
        fields_f32_16[idx].insert(pdat.fields_f32_16[idx]);
    }



    for(u32 idx = 0; idx < pdl.fields_f64.size(); idx++){
        fields_f64[idx].insert(pdat.fields_f64[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_2.size(); idx++){
        fields_f64_2[idx].insert(pdat.fields_f64_2[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_3.size(); idx++){
        fields_f64_3[idx].insert(pdat.fields_f64_3[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_4.size(); idx++){
        fields_f64_4[idx].insert(pdat.fields_f64_4[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_8.size(); idx++){
        fields_f64_8[idx].insert(pdat.fields_f64_8[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_16.size(); idx++){
        fields_f64_16[idx].insert(pdat.fields_f64_16[idx]);
    }





    for(u32 idx = 0; idx < pdl.fields_u32.size(); idx++){
        fields_u32[idx].insert(pdat.fields_u32[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_u64.size(); idx++){
        fields_u64[idx].insert(pdat.fields_u64[idx]);
    }

}



void PatchData::resize(u32 new_obj_cnt){
    for(u32 idx = 0; idx < pdl.fields_f32.size(); idx++){
        fields_f32[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_2.size(); idx++){
        fields_f32_2[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_3.size(); idx++){
        fields_f32_3[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_4.size(); idx++){
        fields_f32_4[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_8.size(); idx++){
        fields_f32_8[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_16.size(); idx++){
        fields_f32_16[idx].resize(new_obj_cnt);
    }



    for(u32 idx = 0; idx < pdl.fields_f64.size(); idx++){
        fields_f64[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_2.size(); idx++){
        fields_f64_2[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_3.size(); idx++){
        fields_f64_3[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_4.size(); idx++){
        fields_f64_4[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_8.size(); idx++){
        fields_f64_8[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_16.size(); idx++){
        fields_f64_16[idx].resize(new_obj_cnt);
    }





    for(u32 idx = 0; idx < pdl.fields_u32.size(); idx++){
        fields_u32[idx].resize(new_obj_cnt);
    }

    for(u32 idx = 0; idx < pdl.fields_u64.size(); idx++){
        fields_u64[idx].resize(new_obj_cnt);
    }
}


void PatchData::append_subset_to(std::vector<u32> & idxs, PatchData &pdat){
    for(u32 idx = 0; idx < pdl.fields_f32.size(); idx++){
        fields_f32[idx].append_subset_to(idxs, pdat.fields_f32[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_2.size(); idx++){
        fields_f32_2[idx].append_subset_to(idxs, pdat.fields_f32_2[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_3.size(); idx++){
        fields_f32_3[idx].append_subset_to(idxs, pdat.fields_f32_3[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_4.size(); idx++){
        fields_f32_4[idx].append_subset_to(idxs, pdat.fields_f32_4[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_8.size(); idx++){
        fields_f32_8[idx].append_subset_to(idxs, pdat.fields_f32_8[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f32_16.size(); idx++){
        fields_f32_16[idx].append_subset_to(idxs, pdat.fields_f32_16[idx]);
    }


    for(u32 idx = 0; idx < pdl.fields_f64.size(); idx++){
        fields_f64[idx].append_subset_to(idxs, pdat.fields_f64[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_2.size(); idx++){
        fields_f64_2[idx].append_subset_to(idxs, pdat.fields_f64_2[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_3.size(); idx++){
        fields_f64_3[idx].append_subset_to(idxs, pdat.fields_f64_3[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_4.size(); idx++){
        fields_f64_4[idx].append_subset_to(idxs, pdat.fields_f64_4[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_8.size(); idx++){
        fields_f64_8[idx].append_subset_to(idxs, pdat.fields_f64_8[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_f64_16.size(); idx++){
        fields_f64_16[idx].append_subset_to(idxs, pdat.fields_f64_16[idx]);
    }


    for(u32 idx = 0; idx < pdl.fields_u32.size(); idx++){
        fields_u32[idx].append_subset_to(idxs, pdat.fields_u32[idx]);
    }

    for(u32 idx = 0; idx < pdl.fields_u64.size(); idx++){
        fields_u64[idx].append_subset_to(idxs, pdat.fields_u64[idx]);
    }
}

template<>
void PatchData::split_patchdata<f32_3>(
    PatchData &pd0, PatchData &pd1, PatchData &pd2, PatchData &pd3, PatchData &pd4, PatchData &pd5, PatchData &pd6, PatchData &pd7, 
    f32_3 bmin_p0, f32_3 bmin_p1, f32_3 bmin_p2, f32_3 bmin_p3, f32_3 bmin_p4, f32_3 bmin_p5, f32_3 bmin_p6, f32_3 bmin_p7, 
    f32_3 bmax_p0, f32_3 bmax_p1, f32_3 bmax_p2, f32_3 bmax_p3, f32_3 bmax_p4, f32_3 bmax_p5, f32_3 bmax_p6, f32_3 bmax_p7){

    u32 field_ipos = pdl.get_field_idx<f32_3>("xyz");
    //TODO check that nvar on this field is 1 on creation

    const u32 obj_cnt = fields_f32_3[field_ipos].size();

    std::vector<u32> idx_p0;
    std::vector<u32> idx_p1;
    std::vector<u32> idx_p2;
    std::vector<u32> idx_p3;
    std::vector<u32> idx_p4;
    std::vector<u32> idx_p5;
    std::vector<u32> idx_p6;
    std::vector<u32> idx_p7;

    for (u32 i = 0; i < obj_cnt; i++) {
    
        f32_3 current_pos = fields_f32_3[field_ipos].usm_data()[i];

        bool bp0 = BBAA::is_particle_in_patch<f32_3>( current_pos, bmin_p0,bmax_p0);
        bool bp1 = BBAA::is_particle_in_patch<f32_3>( current_pos, bmin_p1,bmax_p1);
        bool bp2 = BBAA::is_particle_in_patch<f32_3>( current_pos, bmin_p2,bmax_p2);
        bool bp3 = BBAA::is_particle_in_patch<f32_3>( current_pos, bmin_p3,bmax_p3);
        bool bp4 = BBAA::is_particle_in_patch<f32_3>( current_pos, bmin_p4,bmax_p4);
        bool bp5 = BBAA::is_particle_in_patch<f32_3>( current_pos, bmin_p5,bmax_p5);
        bool bp6 = BBAA::is_particle_in_patch<f32_3>( current_pos, bmin_p6,bmax_p6);
        bool bp7 = BBAA::is_particle_in_patch<f32_3>( current_pos, bmin_p7,bmax_p7);

        if(bp0) idx_p0.push_back(i);
        if(bp1) idx_p1.push_back(i);
        if(bp2) idx_p2.push_back(i);
        if(bp3) idx_p3.push_back(i);
        if(bp4) idx_p4.push_back(i);
        if(bp5) idx_p5.push_back(i);
        if(bp6) idx_p6.push_back(i);
        if(bp7) idx_p7.push_back(i);
    }

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

    const u32 obj_cnt = fields_f64_3[field_ipos].size();

    std::vector<u32> idx_p0;
    std::vector<u32> idx_p1;
    std::vector<u32> idx_p2;
    std::vector<u32> idx_p3;
    std::vector<u32> idx_p4;
    std::vector<u32> idx_p5;
    std::vector<u32> idx_p6;
    std::vector<u32> idx_p7;

    for (u32 i = 0; i < obj_cnt; i++) {
    
        f64_3 current_pos = fields_f64_3[field_ipos].usm_data()[i];

        bool bp0 = BBAA::is_particle_in_patch<f64_3>( current_pos, bmin_p0,bmax_p0);
        bool bp1 = BBAA::is_particle_in_patch<f64_3>( current_pos, bmin_p1,bmax_p1);
        bool bp2 = BBAA::is_particle_in_patch<f64_3>( current_pos, bmin_p2,bmax_p2);
        bool bp3 = BBAA::is_particle_in_patch<f64_3>( current_pos, bmin_p3,bmax_p3);
        bool bp4 = BBAA::is_particle_in_patch<f64_3>( current_pos, bmin_p4,bmax_p4);
        bool bp5 = BBAA::is_particle_in_patch<f64_3>( current_pos, bmin_p5,bmax_p5);
        bool bp6 = BBAA::is_particle_in_patch<f64_3>( current_pos, bmin_p6,bmax_p6);
        bool bp7 = BBAA::is_particle_in_patch<f64_3>( current_pos, bmin_p7,bmax_p7);

        if(bp0) idx_p0.push_back(i);
        if(bp1) idx_p1.push_back(i);
        if(bp2) idx_p2.push_back(i);
        if(bp3) idx_p3.push_back(i);
        if(bp4) idx_p4.push_back(i);
        if(bp5) idx_p5.push_back(i);
        if(bp6) idx_p6.push_back(i);
        if(bp7) idx_p7.push_back(i);
    }

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