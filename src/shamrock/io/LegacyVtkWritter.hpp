// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/collective/io.hpp"
#include "shamalgs/memory/bufferFlattening.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shambase/endian.hpp"
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

        template<class T>
        using repr_t = typename shambase::sycl_utils::VectorProperties<T>::component_type;

        template<class T>
        static constexpr u32 repr_count = shambase::sycl_utils::VectorProperties<T>::dimension; 
        
        template<class T>
        inline repr_t<T>* write_buffer(MPI_File fh,sycl::buffer<T> & buf,u32 len, bool device_alloc ,  u64 & file_head_ptr){

            constexpr u32 new_cnt = len*repr_count<T>;
            constexpr u64 byte_cnt = new_cnt*sizeof(repr_t<T>);

            sycl::buffer<repr_t<T>> buf_w = shamalgs::memory::flatten_buffer(buf,len);
            
            repr_t<T>* usm_buf;
            if(device_alloc){
                usm_buf = sycl::malloc_device<repr_t<T>>(byte_cnt,shamsys::instance::get_compute_queue());
                
            }else{
                usm_buf = sycl::malloc_host<repr_t<T>>(byte_cnt,shamsys::instance::get_compute_queue());
            }


            shamalgs::collective::viewed_write_all_fetch(fh, usm_buf, new_cnt, file_head_ptr);


            sycl::free(usm_buf, shamsys::instance::get_compute_queue());
            
        }

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