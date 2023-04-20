// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/collective/io.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/legacy/log.hpp"
#include <string>

namespace shamrock {

    class LegacyVtkWritter {
        MPI_File mfile{};
        std::string fname;
        bool binary;

        u64 file_head_ptr;

        private:

        inline void head_write(std::string s){
            shamalgs::collective::write_header(mfile , s, file_head_ptr);
        }


        public:

        inline LegacyVtkWritter(std::string fname, bool binary)
            : fname(fname), binary(binary), file_head_ptr(0_u64) {
            logger::debug_ln("VtkWritter", "opening :", fname);
            int rc = mpi::file_open(
                MPI_COMM_WORLD,
                fname.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL,
                &mfile
            );

            if (rc) {
                logger::err_ln("VtkWritter", "Unable to open file :", fname);
            }

            if (binary) {
                head_write("# vtk DataFile Version 5.1\nvtk output\nBINARY\n");
            } else {
                head_write("# vtk DataFile Version 5.1\nvtk output\nASCII\n");
            }
        }

        

        inline ~LegacyVtkWritter() { mpi::file_close(&mfile); }

        LegacyVtkWritter(const LegacyVtkWritter&) = delete;
        LegacyVtkWritter& operator=(const LegacyVtkWritter&) = delete;
        LegacyVtkWritter(LegacyVtkWritter &&other) = delete; // move constructor
        LegacyVtkWritter &operator=(LegacyVtkWritter &&other) =delete; // move assignment
    };
} // namespace shamrock