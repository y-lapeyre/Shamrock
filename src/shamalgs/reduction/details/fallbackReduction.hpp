#pragma once

#include "aliases.hpp"

namespace shamalgs::reduction::details {

    template<class T>
    struct FallbackReduction{

        static T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    };

} // namespace shamalgs::reduction::details