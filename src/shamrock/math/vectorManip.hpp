#pragma once

#include "aliases.hpp"

namespace shamrock::math::vec_manip {

    template<class T>
    struct VectorProperties{
        using component_type = std::void_t<>;
        static constexpr u32 dimension = 0;
    };

    template<class T, u32 dim>
    struct VectorProperties<sycl::vec<T,dim>>{
        using component_type = T;
        static constexpr u32 dimension = dim;
    };

    

} // namespace shamrock::math::vec_manip