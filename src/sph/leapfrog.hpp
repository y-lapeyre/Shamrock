#pragma once

#include "aliases.hpp"
#include "hipSYCL/sycl/buffer.hpp"




template <class flt,class DataLayoutU3>
inline void leapfrog_predictor(sycl::queue &queue, u32 npart, flt dt, 
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_xyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_U3) {

    using vec3 = sycl::vec<flt, 3>;

    sycl::range<1> range_npart{npart};

    auto ker_predict_step = [&](sycl::handler &cgh) {
        auto xyz = buf_xyz->get_access<sycl::access::mode::read_write>(cgh);
        auto U3  = buf_U3->get_access<sycl::access::mode::read_write>(cgh);

        constexpr u32 nvar_U3 = DataLayoutU3::nvar;
        constexpr u32 ivxyz = DataLayoutU3::ivxyz;
        constexpr u32 iaxyz = DataLayoutU3::iaxyz;

        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id();

            vec3 & vxyz = U3[gid*nvar_U3 + ivxyz];
            vec3 & axyz = U3[gid*nvar_U3 + iaxyz];

            // v^{n + 1/2} = v^n + dt/2 a^n
            vxyz = vxyz + (dt / 2) * (axyz);

            // r^{n + 1} = r^n + dt v^{n + 1/2}
            xyz[gid] = xyz[gid] + dt * vxyz;

            // v^* = v^{n + 1/2} + dt/2 a^n
            vxyz = vxyz + (dt / 2) * (axyz);
        });
    };

    queue.submit(ker_predict_step);
}

template <class flt,class DataLayoutU3>
inline void leapfrog_corrector(sycl::queue &queue, u32 npart, flt dt, 
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_xyz,
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &buf_U3){

        sycl::range<1> range_npart{npart};

        using vec3 = sycl::vec<flt, 3>;

        auto ker_corect_step = [&](sycl::handler &cgh) {
                

            auto U3  = buf_U3->get_access<sycl::access::mode::read_write>(cgh);

            constexpr u32 nvar_U3 = DataLayoutU3::nvar;
            constexpr u32 ivxyz = DataLayoutU3::ivxyz;
            constexpr u32 iaxyz = DataLayoutU3::iaxyz;
            constexpr u32 iaxyz_old = DataLayoutU3::iaxyz_old;


            // Executing kernel
            cgh.parallel_for(
                range_npart, [=](sycl::item<1> item) {
                    
                    u32 gid = (u32) item.get_id();

                    vec3 & vxyz = U3[gid*nvar_U3 + ivxyz];
                    vec3 & axyz = U3[gid*nvar_U3 + iaxyz];
                    vec3 & axyz_old = U3[gid*nvar_U3 + iaxyz_old];
        
                    //v^* = v^{n + 1/2} + dt/2 a^n
                    vxyz = vxyz + (dt/2) * (axyz - axyz_old);

                }
            );

        };

        queue.submit(ker_corect_step);

    }
