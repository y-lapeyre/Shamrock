#pragma once

#include "../aliases.hpp"
#include "../flags.hpp"
#include "../sys/mpi_handler.hpp"
#include "../sys/sycl_mpi_interop.hpp"
#include <mpi.h>
#include <vector>

namespace patchdata_layout {

    inline u32 nVarpos_s;
    inline u32 nVarpos_d;
    inline u32 nVarU1_s;
    inline u32 nVarU1_d;
    inline u32 nVarU3_s;
    inline u32 nVarU3_d;

    /**
     * @brief should be check if true before communication with patchdata_s
     */
    inline bool layout_synced = false;

    inline void sync(MPI_Comm comm) {
        mpi::bcast(&patchdata_layout::nVarpos_s, 1, mpi_type_u32, 0, comm);
        mpi::bcast(&patchdata_layout::nVarpos_d, 1, mpi_type_u32, 0, comm);
        mpi::bcast(&patchdata_layout::nVarU1_s , 1, mpi_type_u32, 0, comm);
        mpi::bcast(&patchdata_layout::nVarU1_d , 1, mpi_type_u32, 0, comm);
        mpi::bcast(&patchdata_layout::nVarU3_s , 1, mpi_type_u32, 0, comm);
        mpi::bcast(&patchdata_layout::nVarU3_d , 1, mpi_type_u32, 0, comm);

        layout_synced = true;
    }

    inline void set(u32 arg_nVarpos_s, u32 arg_nVarpos_d, u32 arg_nVarU1_s, u32 arg_nVarU1_d, u32 arg_nVarU3_s,
                    u32 arg_nVarU3_d) {

        patchdata_layout::nVarpos_s = arg_nVarpos_s;
        patchdata_layout::nVarpos_d = arg_nVarpos_d;
        patchdata_layout::nVarU1_s  = arg_nVarU1_s;
        patchdata_layout::nVarU1_d  = arg_nVarU1_d;
        patchdata_layout::nVarU3_s  = arg_nVarU3_s;
        patchdata_layout::nVarU3_d  = arg_nVarU3_d;
    }

    inline bool is_synced(){
        return layout_synced;
    }

} // namespace patchdata_layout



class PatchData {
  public:
    std::vector<f3_s> pos_s;
    std::vector<f3_d> pos_d;
    std::vector<f_s>  U1_s;
    std::vector<f_d>  U1_d;
    std::vector<f3_s> U3_s;
    std::vector<f3_d> U3_d;
};



inline void patchdata_isend(PatchData &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm) {
    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.pos_s.data(), p.pos_s.size(), mpi_type_f3_s, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.pos_d.data(), p.pos_d.size(), mpi_type_f3_d, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.U1_s.data(), p.U1_s.size(), mpi_type_f_s, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.U1_d.data(), p.U1_d.size(), mpi_type_f_d, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.U3_s.data(), p.U3_s.size(), mpi_type_f3_s, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);

    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.U3_d.data(), p.U3_d.size(), mpi_type_f3_d, rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
}

inline PatchData patchdata_irecv( std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm) {
    PatchData p;

    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f3_s, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        p.pos_s.resize(cnt);
        mpi::irecv(p.pos_s.data(), cnt, mpi_type_f3_s, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }

    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f3_d, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        p.pos_d.resize(cnt);
        mpi::irecv(p.pos_d.data(), cnt, mpi_type_f3_d, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f_s, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        p.U1_s.resize(cnt);
        mpi::irecv(p.U1_s.data(), cnt, mpi_type_f_s, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f_d, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        p.U1_d.resize(cnt);
        mpi::irecv(p.U1_d.data(), cnt, mpi_type_f_d, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }




    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f3_s, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        p.U3_s.resize(cnt);
        mpi::irecv(p.U3_s.data(), cnt, mpi_type_f3_s, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    {
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, mpi_type_f3_d, &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        p.U3_d.resize(cnt);
        mpi::irecv(p.U3_d.data(), cnt, mpi_type_f3_d, rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
    }


    return p;
}

