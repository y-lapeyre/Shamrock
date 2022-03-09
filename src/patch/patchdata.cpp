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

#include <exception>
#include <stdexcept>







void patchdata_layout::sync(MPI_Comm comm) {
    mpi::bcast(&patchdata_layout::nVarpos_s, 1, mpi_type_u32, 0, comm);
    mpi::bcast(&patchdata_layout::nVarpos_d, 1, mpi_type_u32, 0, comm);
    mpi::bcast(&patchdata_layout::nVarU1_s , 1, mpi_type_u32, 0, comm);
    mpi::bcast(&patchdata_layout::nVarU1_d , 1, mpi_type_u32, 0, comm);
    mpi::bcast(&patchdata_layout::nVarU3_s , 1, mpi_type_u32, 0, comm);
    mpi::bcast(&patchdata_layout::nVarU3_d , 1, mpi_type_u32, 0, comm);

    layout_synced = true;
}

void patchdata_layout::set(u32 arg_nVarpos_s, u32 arg_nVarpos_d, u32 arg_nVarU1_s, u32 arg_nVarU1_d, u32 arg_nVarU3_s,
                u32 arg_nVarU3_d) {

    if(arg_nVarpos_s + arg_nVarpos_d != 1) 
        throw std::runtime_error("nVarpos_s + nVarpos_d should be equal to 1");

    patchdata_layout::nVarpos_s = arg_nVarpos_s;
    patchdata_layout::nVarpos_d = arg_nVarpos_d;
    patchdata_layout::nVarU1_s  = arg_nVarU1_s;
    patchdata_layout::nVarU1_d  = arg_nVarU1_d;
    patchdata_layout::nVarU3_s  = arg_nVarU3_s;
    patchdata_layout::nVarU3_d  = arg_nVarU3_d;


}

//TODO add check in function calling this one with throw runtimer error
bool patchdata_layout::is_synced(){
    return layout_synced;
}






void patchdata_isend(PatchData &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm) {
    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.pos_s.data(), p.pos_s.size(), mpi_type_f32_3, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.pos_d.data(), p.pos_d.size(), mpi_type_f64_3, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.U1_s.data(), p.U1_s.size(), mpi_type_f32, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.U1_d.data(), p.U1_d.size(), mpi_type_f64, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.U3_s.data(), p.U3_s.size(), mpi_type_f32_3, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.U3_d.data(), p.U3_d.size(), mpi_type_f64_3, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
}

void patchdata_irecv(PatchData & pdat, std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){

    {
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_3, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        pdat.pos_s.resize(cnt);
        mpi::irecv(pdat.pos_s.data(), cnt, mpi_type_f32_3, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_3, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        pdat.pos_d.resize(cnt);
        mpi::irecv(pdat.pos_d.data(), cnt, mpi_type_f64_3, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        pdat.U1_s.resize(cnt);
        mpi::irecv(pdat.U1_s.data(), cnt, mpi_type_f32, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        pdat.U1_d.resize(cnt);
        mpi::irecv(pdat.U1_d.data(), cnt, mpi_type_f64, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }




    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f32_3, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        pdat.U3_s.resize(cnt);
        mpi::irecv(pdat.U3_s.data(), cnt, mpi_type_f32_3, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f64_3, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        pdat.U3_d.resize(cnt);
        mpi::irecv(pdat.U3_d.data(), cnt, mpi_type_f64_3, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


}


PatchData patchdata_gen_dummy_data(std::mt19937& eng){

    std::uniform_int_distribution<u64> distu64(1,1000);

    std::uniform_real_distribution<f64> distfd(-1e5,1e5);

    u32 num_part = distu64(eng);

    PatchData d;


    for (u32 i = 0 ; i < num_part; i++) {
        for (u32 ii = 0; ii < patchdata_layout::nVarpos_s; ii ++) {
            d.pos_s.push_back( f32_3{distfd(eng),distfd(eng),distfd(eng)} );
        }
        
        for (u32 ii = 0; ii < patchdata_layout::nVarpos_d; ii ++) {
            d.pos_d.push_back( f64_3{distfd(eng),distfd(eng),distfd(eng)} );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU1_s; ii ++) {
            d.U1_s.push_back( f32(distfd(eng)) );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU1_d; ii ++) {
            d.U1_d.push_back( f64(distfd(eng)) );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU3_s; ii ++) {
            d.U3_s.push_back( f32_3{distfd(eng),distfd(eng),distfd(eng)} );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU3_d; ii ++) {
            d.U3_d.push_back( f64_3{distfd(eng),distfd(eng),distfd(eng)} );
        }
    }

    return d;
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