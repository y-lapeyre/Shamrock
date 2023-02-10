
#include "aliases.hpp"

namespace shamalgs::memory::details {

    template<class T>
    struct AvoidCopy{

        static T extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx);

    };
    

    


}