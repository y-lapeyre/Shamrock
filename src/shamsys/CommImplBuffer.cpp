#include "CommImplBuffer.hpp"
#include "shamsys/legacy/log.hpp"


namespace shamsys::comm::details {






    //template<class T, Protocol comm_mode> 
    //class CommRequest<sycl::buffer<T>,comm_mode>{
//
    //};


    template class CommBuffer<sycl::buffer<f32   >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f32_2 >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f32_3 >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f32_4 >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f32_8 >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f32_16>,CopyToHost>;
    template class CommBuffer<sycl::buffer<f64   >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f64_2 >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f64_3 >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f64_4 >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f64_8 >,CopyToHost>;
    template class CommBuffer<sycl::buffer<f64_16>,CopyToHost>;
    template class CommBuffer<sycl::buffer<u32   >,CopyToHost>;
    template class CommBuffer<sycl::buffer<u64   >,CopyToHost>;

    template class CommBuffer<sycl::buffer<f32   >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f32_2 >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f32_3 >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f32_4 >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f32_8 >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f32_16>,DirectGPU>;
    template class CommBuffer<sycl::buffer<f64   >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f64_2 >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f64_3 >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f64_4 >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f64_8 >,DirectGPU>;
    template class CommBuffer<sycl::buffer<f64_16>,DirectGPU>;
    template class CommBuffer<sycl::buffer<u32   >,DirectGPU>;
    template class CommBuffer<sycl::buffer<u64   >,DirectGPU>;

    template class CommBuffer<sycl::buffer<f32   >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f32_2 >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f32_3 >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f32_4 >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f32_8 >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f32_16>,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f64   >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f64_2 >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f64_3 >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f64_4 >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f64_8 >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<f64_16>,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<u32   >,DirectGPUFlatten>;
    template class CommBuffer<sycl::buffer<u64   >,DirectGPUFlatten>;
    
} // namespace shamsys::comm::details