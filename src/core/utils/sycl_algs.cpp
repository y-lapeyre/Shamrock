#include "sycl_algs.hpp"
#include "CL/sycl/accessor.hpp"
#include "core/patch/base/enabled_fields.hpp"

template<class T>
class SyclAlg_CopyBuf;

template<class T>
class SyclAlg_CopyBufDiscard;

template<class T>
class SyclAlg_AddWithFactor;

template<class T> class SyclAlg_write_with_offset_into;

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
        bool is_all_true(sycl::buffer<u8> & buf){

            //TODO do it on GPU pleeeaze

            bool res = true;
            {
                sycl::host_accessor acc{buf};

                for (u32 i = 0; i < buf.size(); i++) { //TODO remove ref to size
                    res = res && (acc[i] != 0);
                }
            }

            return res;

        } 
    }

}



#define X(arg)\
template void syclalgs::basic::copybuf_discard<arg>(sycl::buffer<arg> & source, sycl::buffer<arg> & dest, u32 cnt);
XMAC_LIST_ENABLED_FIELD
#undef X

#define X(arg)\
template void syclalgs::basic::copybuf<arg>(sycl::buffer<arg> & source, sycl::buffer<arg> & dest, u32 cnt);
XMAC_LIST_ENABLED_FIELD
#undef X

#define X(arg)\
template void syclalgs::basic::add_with_factor_to<arg>(sycl::buffer<arg> & buf, arg factor, sycl::buffer<arg> & op, u32 cnt);
XMAC_LIST_ENABLED_FIELD
#undef X

#define X(arg)\
template void syclalgs::basic::write_with_offset_into(sycl::buffer<arg> &buf_ctn, sycl::buffer<arg> &buf_in, u32 offset, u32 element_count);
XMAC_LIST_ENABLED_FIELD
#undef X