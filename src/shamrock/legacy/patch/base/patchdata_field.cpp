// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


//%Impl status : Good

#include "patchdata_field.hpp"

#include "aliases.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
//#include "shamrock/legacy/patch/base/pdat_comm_impl/pdat_comm_cp_to_host.hpp"
//#include "shamrock/legacy/patch/base/pdat_comm_impl/pdat_comm_directgpu.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamrock/legacy/algs/sycl/sycl_algs.hpp"
#include <cstdio>
#include <memory>

//TODO use hash for name + nvar to check if the field match before doing operation on them


namespace patchdata_field {

    comm_type current_mode = CopyToHost;

    template <class T>
    PatchDataFieldMpiRequest<T>::PatchDataFieldMpiRequest(PatchDataField<T> &pdat_field, comm_type comm_mode,
                                                          op_type comm_op, u32 comm_val_cnt)
        : comm_mode(comm_mode), comm_op(comm_op), comm_val_cnt(comm_val_cnt), pdat_field(pdat_field) {

        logger::debug_mpi_ln("PatchDataField MPI Comm", "starting mpi sycl comm ", comm_val_cnt, comm_op, comm_mode);

        if (comm_mode == CopyToHost && comm_op == Send) {

            comm_ptr = impl::copy_to_host::send::init<T>(pdat_field.get_buf_priviledge(), comm_val_cnt);

        } else if (comm_mode == CopyToHost && comm_op == Recv_Probe) {

            comm_ptr = impl::copy_to_host::recv::init<T>(comm_val_cnt);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            comm_ptr = impl::directgpu::send::init<T>(pdat_field.get_buf_priviledge(), comm_val_cnt);

        } else if (comm_mode == DirectGPU && comm_op == Recv_Probe) {

            comm_ptr = impl::directgpu::recv::init<T>(comm_val_cnt);

        } else {
            logger::err_ln("PatchDataField MPI Comm", "communication mode & op combination not implemented :", comm_mode,
                           comm_op);
        }
    }

    template <class T> void PatchDataFieldMpiRequest<T>::finalize() {

        logger::debug_mpi_ln("PatchDataField MPI Comm", "finalizing mpi sycl comm ", comm_val_cnt, comm_op, comm_mode);

        if(comm_op == Recv_Probe){
            pdat_field.resize(comm_val_cnt/pdat_field.get_nvar());
        }
        
        if (comm_mode == CopyToHost && comm_op == Send) {

            impl::copy_to_host::send::finalize<T>(comm_ptr);

        } else if (comm_mode == CopyToHost && comm_op == Recv_Probe) {
            impl::copy_to_host::recv::finalize<T>(pdat_field.get_buf_priviledge(), comm_ptr,comm_val_cnt);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            impl::directgpu::send::finalize<T>(comm_ptr);

        } else if (comm_mode == DirectGPU && comm_op == Recv_Probe) {

            impl::copy_to_host::recv::finalize<T>(pdat_field.get_buf_priviledge(), comm_ptr,comm_val_cnt);

        } else {
            logger::err_ln("PatchDataField MPI Comm", "communication mode & op combination not implemented :", comm_mode,
                           comm_op);
        }
    }

#define X(a) template struct PatchDataFieldMpiRequest<a>;
    XMAC_LIST_ENABLED_FIELD
#undef X

} // namespace patchdata_field
