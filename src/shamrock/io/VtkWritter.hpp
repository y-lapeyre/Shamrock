// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include <string>
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/legacy/log.hpp"

namespace shamrock {    


    class VtkWritter{
        MPI_File mfile;
        std::string fname;
        bool binary;

        private:

        inline void write_head(std::string s){
            MPI_Status st;
            if(shamsys::instance::world_rank == 0){
                mpi::file_write(mfile, s.c_str(), s.size(), get_mpi_type<char>(), &st);
            }
        }

        public:


        inline VtkWritter(std::string fname, bool binary): fname(fname), binary(binary){
            logger::debug_ln("VtkWritter", "opening :",fname);
            int rc = mpi::file_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &mfile);

            if (rc) {
                logger::err_ln( "VtkWritter","Unable to open file :", fname);
            }

            if(binary){
                write_head("# vtk DataFile Version 5.1\nvtk output\nBINARY\n");
            }else{
                write_head("# vtk DataFile Version 5.1\nvtk output\nASCII\n");
            }
            
        }


        inline ~VtkWritter(){
            mpi::file_close(&mfile);
        }



    };
}