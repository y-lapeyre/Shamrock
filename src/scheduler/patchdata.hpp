#pragma once

#include "../aliases.hpp"
#include "../sys/mpi_handler.hpp"
#include "../sys/sycl_mpi_interop.hpp"
#include <mpi.h>

namespace patchdata_layout {
inline u32 nVarpos_s;
inline u32 nVarpos_d;
inline u32 nVarU1_s;
inline u32 nVarU1_d;
inline u32 nVarU3_s;
inline u32 nVarU3_d;

inline void sync(MPI_Comm comm) {
  mpi::bcast(&patchdata_layout::nVarpos_s, 1, mpi_type_u32, 0, comm);
  mpi::bcast(&patchdata_layout::nVarpos_d, 1, mpi_type_u32, 0, comm);
  mpi::bcast(&patchdata_layout::nVarU1_s, 1, mpi_type_u32, 0, comm);
  mpi::bcast(&patchdata_layout::nVarU1_d, 1, mpi_type_u32, 0, comm);
  mpi::bcast(&patchdata_layout::nVarU3_s, 1, mpi_type_u32, 0, comm);
  mpi::bcast(&patchdata_layout::nVarU3_d, 1, mpi_type_u32, 0, comm);
}

inline void set(u32 arg_nVarpos_s, u32 arg_nVarpos_d, u32 arg_nVarU1_s,
                u32 arg_nVarU1_d, u32 arg_nVarU3_s, u32 arg_nVarU3_d) {

  patchdata_layout::nVarpos_s = arg_nVarpos_s;
  patchdata_layout::nVarpos_d = arg_nVarpos_d;
  patchdata_layout::nVarU1_s = arg_nVarU1_s;
  patchdata_layout::nVarU1_d = arg_nVarU1_d;
  patchdata_layout::nVarU3_s = arg_nVarU3_s;
  patchdata_layout::nVarU3_d = arg_nVarU3_d;
}

} // namespace patchdata_layout
