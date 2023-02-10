
#include "aliases.hpp"

namespace shamalgs::memory::details {

    template<class T>
    struct Fallback{

        static T extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx);

        
        static sycl::buffer<T> vec_to_buf(const std::vector<T> &vec);
        static std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len);

    };
    

    


}