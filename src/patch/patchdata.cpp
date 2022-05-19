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
#include "patch/patchdata_field.hpp"
#include "sys/sycl_mpi_interop.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>
#include <vector>







void patchdata_isend(PatchData &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm) {

    for (auto a : p.fields_f32) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f32, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f32_2) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f32_2, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f32_3) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f32_3, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f32_4) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f32_4, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f32_8) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f32_8, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f32_16) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f32_16, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    for (auto a : p.fields_f64) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f64, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f64_2) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f64_2, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f64_3) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f64_3, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f64_4) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f64_4, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f64_8) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f64_8, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_f64_16) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_f64_16, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_u32) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_u32, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto a : p.fields_u64) {
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(a.data(), a.size(), mpi_type_u64, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }
}

void patchdata_irecv(PatchData & pdat, std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){

    for (auto & a : pdat.fields_f32) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    for (auto & a : pdat.fields_f32_2) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_2, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_2, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f32_3) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_3, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_3, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f32_4) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_4, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_4, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f32_8) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_8, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_8, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f32_16) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_16, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f32_16, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }





    for (auto & a : pdat.fields_f64) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    for (auto & a : pdat.fields_f64_2) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_2, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_2, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f64_3) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_3, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_3, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f64_4) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_4, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_4, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f64_8) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_8, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_8, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_f64_16) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_16, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_f64_16, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }
    



    for (auto & a : pdat.fields_u32) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_u32, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_u32, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    for (auto & a : pdat.fields_u64) {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_u64, &cnt);
        rq_lst.resize(rq_lst.size() + 1);

        u32 len = cnt / a.get_nvar();

        a.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(a.data(), cnt, mpi_type_u64, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


}


PatchData patchdata_gen_dummy_data(PatchDataLayout & pdl, std::mt19937& eng){

    std::uniform_int_distribution<u64> distu64(1,1000);

    std::uniform_real_distribution<f64> distfd(-1e5,1e5);

    u32 num_part = distu64(eng);

    PatchData pdat(pdl);


    for (auto & a : pdat.fields_f32) {
        a.resize(num_part);
    }


    for (auto & a : pdat.fields_f32_2) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f32_3) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f32_4) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f32_8) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f32_16) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f64) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f64_2) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f64_3) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f64_4) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f64_8) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_f64_16) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_u32) {
        a.resize(num_part);
    }

    for (auto & a : pdat.fields_u64) {
        a.resize(num_part);
    }




    for (u32 i = 0 ; i < num_part; i++) {
        for (auto & a : pdat.fields_f32) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f32(distfd(eng));
        }


        for (auto & a : pdat.fields_f32_2) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f32_2{distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f32_3) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f32_3{distfd(eng),distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f32_4) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f32_4{distfd(eng),distfd(eng),distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f32_8) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f32_8{distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f32_16) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f32_16{distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f64) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f64(distfd(eng));
        }

        for (auto & a : pdat.fields_f64_2) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f64_2{distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f64_3) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f64_3{distfd(eng),distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f64_4) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f64_4{distfd(eng),distfd(eng),distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f64_8) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f64_8{distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_f64_16) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = f64_16{distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng),distfd(eng)};
        }

        for (auto & a : pdat.fields_u32) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = u32(distu64(eng));
        }

        for (auto & a : pdat.fields_u64) {
            for(u32 i = 0 ; i < a.get_nvar(); i++) a.data()[i*a.get_nvar()] = u64(distu64(eng));
        }
    }

    return pdat;
}

template<class T>
bool check_field_match(PatchDataField<T> &f1, PatchDataField<T> &f2){
    
}

bool patch_data_check_match(PatchData& p1, PatchData& p2){
    bool check = true;


    check = check && ( p1.pos_s.size() == p2.pos_s.size());
    check = check && ( p1.pos_d.size() == p2.pos_d.size());
    check = check && ( p1.U1_s.size()  == p2.U1_s.size() );
    check = check && ( p1.U1_d.size()  == p2.U1_d.size() );
    check = check && ( p1.U3_s.size()  == p2.U3_s.size() );
    check = check && ( p1.U3_d.size()  == p2.U3_d.size() );


    for (u32 i = 0; i < p1.pos_s.size(); i ++) {
        check = check && (test_eq3(p1.pos_s[i] , p2.pos_s[i] ));
    }
    
    for (u32 i = 0; i < p1.pos_d.size(); i ++) {
        check = check && (test_eq3(p1.pos_d[i] , p2.pos_d[i] ));
    }

    for (u32 i = 0; i < p1.U1_s.size(); i ++) {
        check = check && (p1.U1_s[i] == p2.U1_s[i] );
    }
    
    for (u32 i = 0; i < p1.U1_d.size(); i ++) {
        check = check && (p1.U1_d[i] == p2.U1_d[i] );
    }

    for (u32 i = 0; i < p1.U3_s.size(); i ++) {
        check = check && (test_eq3(p1.U3_s[i] , p2.U3_s[i] ));
    }
    
    for (u32 i = 0; i < p1.U3_d.size(); i ++) {
        check = check && (test_eq3(p1.U3_d[i] , p2.U3_d[i] ));
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

void PatchData::extract_particle(u32 pidx, 
    std::vector<f32_3> &out_pos_s, 
    std::vector<f64_3> &out_pos_d, 
    std::vector<f32> &out_U1_s, 
    std::vector<f64> &out_U1_d, 
    std::vector<f32_3> &out_U3_s, 
    std::vector<f64_3> &out_U3_d){

    const u64 idx_pos_s = pidx * patchdata_layout::nVarpos_s;
    const u64 idx_pos_d = pidx * patchdata_layout::nVarpos_d;
    const u64 idx_U1_s  = pidx * patchdata_layout::nVarU1_s ;
    const u64 idx_U1_d  = pidx * patchdata_layout::nVarU1_d ;
    const u64 idx_U3_s  = pidx * patchdata_layout::nVarU3_s ;
    const u64 idx_U3_d  = pidx * patchdata_layout::nVarU3_d ;

    const u64 idx_out_pos_s = out_pos_s.size();
    const u64 idx_out_pos_d = out_pos_d.size();
    const u64 idx_out_U1_s  = out_U1_s .size();
    const u64 idx_out_U1_d  = out_U1_d .size();
    const u64 idx_out_U3_s  = out_U3_s .size();
    const u64 idx_out_U3_d  = out_U3_d .size();

    out_pos_s.resize(idx_out_pos_s + patchdata_layout::nVarpos_s);
    out_pos_d.resize(idx_out_pos_d + patchdata_layout::nVarpos_d);
    out_U1_s .resize(idx_out_U1_s  + patchdata_layout::nVarU1_s );
    out_U1_d .resize(idx_out_U1_d  + patchdata_layout::nVarU1_d );
    out_U3_s .resize(idx_out_U3_s  + patchdata_layout::nVarU3_s );
    out_U3_d .resize(idx_out_U3_d  + patchdata_layout::nVarU3_d );

    for(u32 i = patchdata_layout::nVarpos_s-1 ; i < patchdata_layout::nVarpos_s ; i--){
        out_pos_s[idx_out_pos_s + i] = (fast_extract(idx_pos_s + i, pos_s));
    }

    for(u32 i = patchdata_layout::nVarpos_d-1 ; i < patchdata_layout::nVarpos_d ; i--){
        out_pos_d[idx_out_pos_d + i] = (fast_extract(idx_pos_d + i, pos_d));
    }

    for(u32 i = patchdata_layout::nVarU1_s-1 ; i < patchdata_layout::nVarU1_s ; i--){
        out_U1_s[idx_out_U1_s + i] = (fast_extract(idx_U1_s + i, U1_s));
    }

    for(u32 i = patchdata_layout::nVarU1_d-1 ; i < patchdata_layout::nVarU1_d ; i--){
        out_U1_d[idx_out_U1_d + i] = (fast_extract(idx_U1_d + i, U1_d));
    }

    for(u32 i = patchdata_layout::nVarU3_s-1 ; i < patchdata_layout::nVarU3_s ; i--){
        out_U3_s[idx_out_U3_s + i] = (fast_extract(idx_U3_s + i, U3_s));
    }

    for(u32 i = patchdata_layout::nVarU3_d-1 ; i < patchdata_layout::nVarU3_d ; i--){
        out_U3_d[idx_out_U3_d + i] = (fast_extract(idx_U3_d + i, U3_d));
    }

}

void PatchData::insert_particles(std::vector<f32_3> &in_pos_s, std::vector<f64_3> &in_pos_d, std::vector<f32> &in_U1_s, std::vector<f64> &in_U1_d, std::vector<f32_3> &in_U3_s, std::vector<f64_3> &in_U3_d){
    pos_s.insert(pos_s.end(),in_pos_s.begin(), in_pos_s.end());
    pos_d.insert(pos_d.end(),in_pos_d.begin(), in_pos_d.end());
    U1_s .insert(U1_s .end(),in_U1_s .begin(), in_U1_s .end());
    U1_d .insert(U1_d .end(),in_U1_d .begin(), in_U1_d .end());
    U3_s .insert(U3_s .end(),in_U3_s .begin(), in_U3_s .end());
    U3_d .insert(U3_d .end(),in_U3_d .begin(), in_U3_d .end());
}