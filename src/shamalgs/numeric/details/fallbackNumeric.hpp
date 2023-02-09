
#include "aliases.hpp"

namespace shamalgs::numeric::details {

    template<class T>
    struct FallbackNumeric {

        static sycl::buffer<T> exclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    };

} // namespace shamalgs::numeric::details
