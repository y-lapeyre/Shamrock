// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shamsys/legacy/log.hpp"
#include "sycl_utils/sycl_utilities.hpp"
#include "sycl_utils/vec_equals.hpp"
#include "sycl_utils/vectorProperties.hpp"
#include <stdexcept>
#include "shambase/stacktrace.hpp"

namespace shambase {

    /**
     * @brief check that the size of a sycl buffer is below or equal to the value of max range
     * throw if it is not the case
     *
     * @tparam T
     * @param buf
     * @param max_range
     * @param loc
     */
    template<class T>
    void check_buffer_size(sycl::buffer<T> &buf,
                           u64 max_range,
                           const SourceLocation loc = SourceLocation()) {
        if (buf.size() < max_range) {
            throw throw_with_loc<std::invalid_argument>("buffer is too small", loc);
        }
    }

    /**
     * @brief Get the Device Type Name
     *
     * @param Device
     * @return std::string
     */
    inline std::string getDevice_type(const sycl::device &Device) {
        auto DeviceType = Device.get_info<sycl::info::device::device_type>();
        switch (DeviceType) {
        case sycl::info::device_type::cpu: return "CPU";
        case sycl::info::device_type::gpu: return "GPU";
        case sycl::info::device_type::host: return "HOST";
        case sycl::info::device_type::accelerator: return "ACCELERATOR";
        default: return "UNKNOWN";
        }
    }

    /**
     * @brief Generate a sycl nd range out of a group size and lenght
     * 
     * @param lenght max index value
     * @param group_size group size
     * @return sycl::nd_range<1> the sycl nd range
     */
    inline sycl::nd_range<1> make_range(u32 lenght, const u32 group_size = 32){
        u32 group_cnt = shambase::group_count(lenght, group_size);
        u32 len =group_cnt*group_size;
        return sycl::nd_range<1>{len, group_size};
    }

    enum ParralelForWrapMode {
        PARRALEL_FOR,PARRALEL_FOR_ROUND, ND_RANGE
    };

    #ifdef SHAMROCK_LOOP_DEFAULT_PARRALEL_FOR
    constexpr ParralelForWrapMode default_loop_mode = PARRALEL_FOR;
    #endif

    #ifdef SHAMROCK_LOOP_DEFAULT_PARRALEL_FOR_ROUND
    constexpr ParralelForWrapMode default_loop_mode = PARRALEL_FOR_ROUND;
    #endif
    
    #ifdef SHAMROCK_LOOP_DEFAULT_ND_RANGE
    constexpr ParralelForWrapMode default_loop_mode = ND_RANGE;
    #endif

    constexpr u32 default_gsize = SHAMROCK_LOOP_GSIZE;
    constexpr u32 default_gsize_2d = 16;
    constexpr u32 default_gsize_3d = 4;


    template<u32 group_size = default_gsize, ParralelForWrapMode mode = default_loop_mode,class LambdaKernel>
    inline void parralel_for(sycl::handler & cgh, u32 lenght, const char* name, LambdaKernel && ker){

        #ifdef SHAMROCK_USE_NVTX
        nvtxRangePush(name);
        #endif

        logger::debug_sycl_ln("SYCL", shambase::format("parralel_for {} N={}",name,lenght));
        
        if constexpr(mode == PARRALEL_FOR){

            cgh.parallel_for(sycl::range<1>{lenght}, [=](sycl::item<1> id){
                ker(id.get_linear_id());
            });

        }else if constexpr(mode == PARRALEL_FOR_ROUND){

            u32 len = shambase::group_count(lenght, group_size)*group_size;

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id){

                u64 gid = id.get_linear_id();
                if(gid >= lenght) return;

                ker(gid);
            });

        }else if constexpr(mode==ND_RANGE){

            cgh.parallel_for(make_range(lenght,group_size), [=](sycl::nd_item<1> id){

                u64 gid = id.get_global_linear_id();
                if(gid >= lenght) return;

                ker(gid);
            });

        }else{
            throw_unimplemented();
        }

        #ifdef SHAMROCK_USE_NVTX
        nvtxRangePop();
        #endif
    }
    
    template<u32 group_size = default_gsize_2d, ParralelForWrapMode mode = default_loop_mode,class LambdaKernel>
    inline void parralel_for_2d(sycl::handler & cgh, u32 lenght_x, u32 lenght_y, const char* name, LambdaKernel && ker){

        #ifdef SHAMROCK_USE_NVTX
        nvtxRangePush(name);
        #endif

        logger::debug_sycl_ln("SYCL", shambase::format("parralel_for {} N={} {}",name,lenght_x,lenght_y));
        
        if constexpr(mode == PARRALEL_FOR){

            cgh.parallel_for(sycl::range<2>{lenght_x,lenght_y}, [=](sycl::item<2> id){
                ker(id.get_id(0),id.get_id(1));
            });

        }else if constexpr(mode == PARRALEL_FOR_ROUND){

            u32 len_x = shambase::group_count(lenght_x, group_size)*group_size;
            u32 len_y = shambase::group_count(lenght_y, group_size)*group_size;

            cgh.parallel_for(sycl::range<2>{len_x,len_y}, [=](sycl::item<2> id){

                if(id.get_id(0) >= lenght_x || id.get_id(1) >= lenght_y) return;

                ker(id.get_id(0),id.get_id(1));
            });

        }else if constexpr(mode==ND_RANGE){

            sycl::nd_range<1> rx = make_range(lenght_x,group_size);
            sycl::nd_range<1> ry = make_range(lenght_y,group_size);

            sycl::range<2> tmp_s {rx.get_global_range().size(),ry.get_global_range().size()};
            sycl::range<2> tmp_g {rx.get_group_range().size(),ry.get_group_range().size()};

            cgh.parallel_for(sycl::nd_range<2>{tmp_s,tmp_g}, [=](sycl::nd_item<2> id){

                if(id.get_global_id(0) >= lenght_x || id.get_global_id(1) >= lenght_y) return;

                ker(id.get_global_id(0),id.get_global_id(1));
            });

        }else{
            throw_unimplemented();
        }

        #ifdef SHAMROCK_USE_NVTX
        nvtxRangePop();
        #endif
    }

    template<u32 group_size = default_gsize_3d, ParralelForWrapMode mode = default_loop_mode,class LambdaKernel>
    inline void parralel_for_3d(sycl::handler & cgh, u32 lenght_x, u32 lenght_y, u32 lenght_z, const char* name, LambdaKernel && ker){

        #ifdef SHAMROCK_USE_NVTX
        nvtxRangePush(name);
        #endif

        logger::debug_sycl_ln("SYCL", shambase::format("parralel_for {} N={} {} {}",name,lenght_x,lenght_y,lenght_z));
        
        if constexpr(mode == PARRALEL_FOR){

            cgh.parallel_for(sycl::range<3>{lenght_x,lenght_y,lenght_z}, [=](sycl::item<3> id){
                ker(id.get_id(0),id.get_id(1),id.get_id(2));
            });

        }else if constexpr(mode == PARRALEL_FOR_ROUND){

            u32 len_x = shambase::group_count(lenght_x, group_size)*group_size;
            u32 len_y = shambase::group_count(lenght_y, group_size)*group_size;
            u32 len_z = shambase::group_count(lenght_z, group_size)*group_size;

            cgh.parallel_for(sycl::range<3>{len_x,len_y,len_z}, [=](sycl::item<3> id){

                if(id.get_id(0) >= lenght_x || id.get_id(1) >= lenght_y || id.get_id(2) >= lenght_z) return;

                ker(id.get_id(0),id.get_id(1),id.get_id(2));
            });

        }else if constexpr(mode==ND_RANGE){

            sycl::nd_range<1> rx = make_range(lenght_x,group_size);
            sycl::nd_range<1> ry = make_range(lenght_y,group_size);
            sycl::nd_range<1> rz = make_range(lenght_z,group_size);

            sycl::range<3> tmp_s {rx.get_global_range().size(),ry.get_global_range().size(),rz.get_global_range().size()};
            sycl::range<3> tmp_g {rx.get_group_range().size(),ry.get_group_range().size(),rz.get_group_range().size()};

            cgh.parallel_for(sycl::nd_range<3>{tmp_s,tmp_g}, [=](sycl::nd_item<3> id){

                if(id.get_global_id(0) >= lenght_x || id.get_global_id(1) >= lenght_y|| id.get_global_id(2) >= lenght_z) return;

                ker(id.get_global_id(0),id.get_global_id(1),id.get_global_id(2));
            });

        }else{
            throw_unimplemented();
        }

        #ifdef SHAMROCK_USE_NVTX
        nvtxRangePop();
        #endif

    }
    

    template<ParralelForWrapMode mode = default_loop_mode,class LambdaKernel>
    inline void parralel_for_gsize(sycl::handler & cgh, u32 lenght, u32 group_size, const char* name, LambdaKernel && ker){

        #ifdef SHAMROCK_USE_NVTX
        nvtxRangePush(name);
        #endif
        
        if constexpr(mode == PARRALEL_FOR){

            cgh.parallel_for(sycl::range<1>{lenght}, [=](sycl::item<1> id){
                ker(id.get_linear_id());
            });

        }else if constexpr(mode == PARRALEL_FOR_ROUND){

            u32 len = shambase::group_count(lenght, group_size)*group_size;

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id){

                u64 gid = id.get_linear_id();
                if(gid >= lenght) return;

                ker(gid);
            });

        }else if constexpr(mode==ND_RANGE){

            cgh.parallel_for(make_range(lenght,group_size), [=](sycl::nd_item<1> id){

                u64 gid = id.get_global_linear_id();
                if(gid >= lenght) return;

                ker(gid);
            });

        }else{
            throw_unimplemented();
        }

        #ifdef SHAMROCK_USE_NVTX
        nvtxRangePop();
        #endif
    }

    inline void check_queue_state(sycl::queue & q
                           , SourceLocation loc = SourceLocation()){
        logger::debug_sycl_ln("SYCL", "checking queue state", loc.format_one_line());
        q.wait_and_throw();
        logger::debug_sycl_ln("SYCL", "checking queue state : OK !");
    }

} // namespace shambase