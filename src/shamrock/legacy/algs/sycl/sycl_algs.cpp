// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sycl_algs.hpp"
#include "shamalgs/memory/memory.hpp"
#include "shamalgs/reduction/reduction.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"

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



        
        
        





        




        

        

    } // namespace basic

    namespace reduction {




        template<class T> bool equals(sycl::buffer<T> &buf1, sycl::buffer<T> &buf2, u32 cnt){

            sycl::buffer<u8> res (cnt);
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                sycl::accessor acc1 {buf1, cgh, sycl::read_only};
                sycl::accessor acc2 {buf2, cgh, sycl::read_only};

                sycl::accessor out {res, cgh, sycl::write_only, sycl::no_init};
                
                cgh.parallel_for<SyclAlg_IsSame<T>>( sycl::range{cnt}, [=](sycl::item<1> item) { 
                    out[item] = test_sycl_eq(acc1[item],acc2[item]); });
            });


            return shamalgs::reduction::is_all_true(res,cnt);

        }




    }


    namespace convert {
        template<class T> 
        sycl::buffer<T> vector_to_buf(std::vector<T> && vec){

            u32 cnt = vec.size();
            sycl::buffer<T> ret(cnt);

            sycl::buffer<T> alias(vec.data(),cnt);

            shamalgs::memory::copybuf_discard(alias, ret, cnt);

            //HIPSYCL segfault otherwise because looks like the destructor of the sycl buffer 
            //doesn't wait for the end of the queue resulting in out of bound access
            #ifdef SYCL_COMP_OPENSYCL
            shamsys::instance::get_compute_queue().wait();
            #endif

            return std::move(ret);

        }

        template<class T> 
        sycl::buffer<T> vector_to_buf(std::vector<T> & vec){

            u32 cnt = vec.size();
            sycl::buffer<T> ret(cnt);

            sycl::buffer<T> alias(vec.data(),cnt);

            shamalgs::memory::copybuf_discard(alias, ret, cnt);

            //HIPSYCL segfault otherwise because looks like the destructor of the sycl buffer 
            //doesn't wait for the end of the queue resulting in out of bound access
            #ifdef SYCL_COMP_OPENSYCL
            shamsys::instance::get_compute_queue().wait();
            #endif

            return std::move(ret);

        }
    } // namespace convert

} // namespace syclalgs


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