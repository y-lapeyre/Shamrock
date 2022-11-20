// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sycl_algs.hpp"
#include "hipSYCL/sycl/libkernel/accessor.hpp"
#include "sycl_vector_utils.hpp"
#include "core/sys/sycl_mpi_interop.hpp"

//%Impl status : Clean


template<class T>
class SyclAlg_CopyBuf;

template<class T>
class SyclAlg_CopyBufDiscard;

template<class T>
class SyclAlg_AddWithFactor;

template<class T> 
class SyclAlg_write_with_offset_into;

template<class T> 
class SyclAlg_IsSame;

namespace syclalgs {

    namespace basic {

        template<class T>
        void copybuf(sycl::buffer<T> & source, sycl::buffer<T> & dest, u32 cnt){
            sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){

                sycl::accessor src {source,cgh,sycl::read_only};
                sycl::accessor dst {dest,cgh,sycl::write_only};

                cgh.parallel_for<SyclAlg_CopyBuf<T>>(sycl::range<1>{cnt},[=](sycl::item<1> i){
                    dst[i] = src[i];
                });

            });
        }


        
        
        template<class T>
        void copybuf_discard(sycl::buffer<T> & source, sycl::buffer<T> & dest, u32 cnt){
            sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){

                sycl::accessor src {source,cgh,sycl::read_only};
                sycl::accessor dst {dest,cgh,sycl::write_only,sycl::no_init};

                cgh.parallel_for<SyclAlg_CopyBufDiscard<T>>(sycl::range<1>{cnt},[=](sycl::item<1> i){
                    dst[i] = src[i];
                });

            });
        }


        template<class T>
        void add_with_factor_to(sycl::buffer<T> & buf, T factor, sycl::buffer<T> & op, u32 cnt){
            sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){

                sycl::accessor acc {buf,cgh,sycl::read_write};
                sycl::accessor dd {op,cgh,sycl::read_only};

                T fac = factor;

                cgh.parallel_for<SyclAlg_AddWithFactor<T>>(sycl::range<1>{cnt},[=](sycl::item<1> i){
                    acc[i] += fac*dd[i];
                });

            });
        }




        

        template<class T>
        void write_with_offset_into(sycl::buffer<T> & buf_ctn, sycl::buffer<T> & buf_in, u32 offset, u32 element_count){
            sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

                sycl::accessor source {buf_in, cgh, sycl::read_only};
                sycl::accessor dest {buf_ctn, cgh, sycl::write_only};
                u32 off = offset;
                cgh.parallel_for<SyclAlg_write_with_offset_into<T>>( sycl::range{element_count}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
            });
        }

        

    }

    namespace reduction {













        bool is_all_true(sycl::buffer<u8> & buf,u32 cnt){

            //TODO do it on GPU pleeeaze

            bool res = true;
            {
                sycl::host_accessor acc{buf, sycl::read_only};

                for (u32 i = 0; i < cnt; i++) { //TODO remove ref to size
                    res = res && (acc[i] != 0);
                }
            }

            return res;

        } 

        template<class T> bool equals(sycl::buffer<T> &buf1, sycl::buffer<T> &buf2, u32 cnt){

            sycl::buffer<u8> res (cnt);
            sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

                sycl::accessor acc1 {buf1, cgh, sycl::read_only};
                sycl::accessor acc2 {buf2, cgh, sycl::read_only};

                sycl::accessor out {res, cgh, sycl::write_only, sycl::no_init};
                
                cgh.parallel_for<SyclAlg_IsSame<T>>( sycl::range{cnt}, [=](sycl::item<1> item) { 
                    out[item] = test_sycl_eq(acc1[item],acc2[item]); });
            });


            return is_all_true(res,cnt);
        }
    }


    namespace convert {
        template<class T> 
        sycl::buffer<T> vector_to_buf(std::vector<T> && vec){

            u32 cnt = vec.size();
            sycl::buffer<T> ret(cnt);

            sycl::buffer<T> alias(vec.data(),cnt);

            basic::copybuf_discard(alias, ret, cnt);

            //HIPSYCL segfault otherwise because looks like the destructor of the sycl buffer 
            //doesn't wait for the end of the queue resulting in out of bound access
            #ifdef SYCL_COMP_HIPSYCL
            sycl_handler::get_compute_queue().wait();
            #endif

            return std::move(ret);

        }

        template<class T> 
        sycl::buffer<T> vector_to_buf(std::vector<T> & vec){

            u32 cnt = vec.size();
            sycl::buffer<T> ret(cnt);

            sycl::buffer<T> alias(vec.data(),cnt);

            basic::copybuf_discard(alias, ret, cnt);

            //HIPSYCL segfault otherwise because looks like the destructor of the sycl buffer 
            //doesn't wait for the end of the queue resulting in out of bound access
            #ifdef SYCL_COMP_HIPSYCL
            sycl_handler::get_compute_queue().wait();
            #endif

            return std::move(ret);

        }
    }

}


#define X(arg)\
template void syclalgs::basic::copybuf_discard<arg>(sycl::buffer<arg> & source, sycl::buffer<arg> & dest, u32 cnt);
XMAC_SYCLMPI_TYPE_ENABLED
#undef X

#define X(arg)\
template void syclalgs::basic::copybuf<arg>(sycl::buffer<arg> & source, sycl::buffer<arg> & dest, u32 cnt);
XMAC_SYCLMPI_TYPE_ENABLED
#undef X

#define X(arg)\
template void syclalgs::basic::add_with_factor_to<arg>(sycl::buffer<arg> & buf, arg factor, sycl::buffer<arg> & op, u32 cnt);
XMAC_SYCLMPI_TYPE_ENABLED
#undef X

#define X(arg)\
template void syclalgs::basic::write_with_offset_into(sycl::buffer<arg> &buf_ctn, sycl::buffer<arg> &buf_in, u32 offset, u32 element_count);
XMAC_SYCLMPI_TYPE_ENABLED
#undef X

#define X(arg)\
template bool syclalgs::reduction::equals(sycl::buffer<arg> &buf1, sycl::buffer<arg> &buf2, u32 cnt);
XMAC_SYCLMPI_TYPE_ENABLED
#undef X


#define X(arg)\
template sycl::buffer<arg> syclalgs::convert::vector_to_buf(std::vector<arg> && vec);
XMAC_SYCLMPI_TYPE_ENABLED
#undef X

#define X(arg)\
template sycl::buffer<arg> syclalgs::convert::vector_to_buf(std::vector<arg> & vec);
XMAC_SYCLMPI_TYPE_ENABLED
#undef X