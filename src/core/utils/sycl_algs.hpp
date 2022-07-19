#pragma once

#include "CL/sycl/access/access.hpp"
#include "aliases.hpp"
#include "core/sys/sycl_handler.hpp"


template<class T>
class SyclAlg_CopyBuf;

template<class T>
class SyclAlg_CopyBufDiscard;


template<class T>
class SyclAlg_AddWithFactor;

namespace syclalgs {

    namespace basic {

        
        

        template<class T>
        inline void copybuf(sycl::buffer<T> & source, sycl::buffer<T> & dest, u32 cnt){
            sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){

                sycl::accessor src {source,cgh,sycl::read_only};
                sycl::accessor dst {dest,cgh,sycl::write_only};

                cgh.parallel_for<SyclAlg_CopyBuf<T>>(sycl::range<1>{cnt},[=](sycl::item<1> i){
                    dst[i] = src[i];
                });

            });
        }


        
        
        template<class T>
        inline void copybuf_discard(sycl::buffer<T> & source, sycl::buffer<T> & dest, u32 cnt){
            sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){

                sycl::accessor src {source,cgh,sycl::read_only};
                sycl::accessor dst {dest,cgh,sycl::write_only,sycl::noinit};

                cgh.parallel_for<SyclAlg_CopyBufDiscard<T>>(sycl::range<1>{cnt},[=](sycl::item<1> i){
                    dst[i] = src[i];
                });

            });
        }


        template<class T>
        inline void add_with_factor_to(sycl::buffer<T> & buf, T factor, sycl::buffer<T> & op, u32 cnt){
            sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){

                sycl::accessor acc {buf,cgh,sycl::read_write};
                sycl::accessor dd {op,cgh,sycl::read_only};

                T fac = factor;

                cgh.parallel_for<SyclAlg_AddWithFactor<T>>(sycl::range<1>{cnt},[=](sycl::item<1> i){
                    acc[i] = dd[i]*fac;
                });

            });
        }

    }

    namespace reduction {
        
    }

}