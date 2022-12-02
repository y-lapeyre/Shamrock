#include "aliases.hpp"
#include "core/sys/sycl_handler.hpp"


//%Impl status : Clean

namespace syclalgs {

    namespace basic {

        template <class T> void copybuf(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

        template <class T> void copybuf_discard(sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt);

        template<class T>
        void write_with_offset_into(sycl::buffer<T> & buf_ctn, sycl::buffer<T> & buf_in, u32 offset, u32 element_count);

    } // namespace basic

    namespace reduction {

        bool is_all_true(sycl::buffer<u8> & buf, u32 cnt);

        template <class T> 
        bool equals(sycl::buffer<T> & buf1, sycl::buffer<T> & buf2, u32 cnt);
        
    } // namespace reduction

    namespace convert {
        template<class T> sycl::buffer<T> vector_to_buf(std::vector<T> && vec);

        template<class T> sycl::buffer<T> vector_to_buf(std::vector<T> & vec);
    } // namespace convert

} // namespace syclalgs


