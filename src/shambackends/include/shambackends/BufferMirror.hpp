// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DeviceBuffer.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */


#include "shambase/SourceLocation.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shamcomm/logs.hpp"
namespace sham {

    template<class T, USMKindTarget target, USMKindTarget orgin_target>
    class BufferMirror{

        DeviceBuffer<T, orgin_target> & mirrored_buffer;

        DeviceBuffer<T, target> mirror;
        T* ptr_mirror;

    public:

        BufferMirror(DeviceBuffer<T, orgin_target> & mirrored_buffer): mirrored_buffer(mirrored_buffer),
        mirror (mirrored_buffer.template copy_to<target>())
        {
            sham::EventList depends_list;
            ptr_mirror = mirror.get_write_access(depends_list);
            depends_list.wait();
            mirror.complete_event_state(sycl::event{});
        }

        BufferMirror(const BufferMirror&) = delete;
        BufferMirror(BufferMirror&&) = delete;
        BufferMirror& operator=(const BufferMirror&) = delete;
        BufferMirror& operator=(BufferMirror&&) = delete;

        T* data() const { return ptr_mirror; }
        u32 size() const { return mirrored_buffer.size(); }

        T& operator[](u32 i) const { return ptr_mirror[i]; }

        ~BufferMirror(){
            mirrored_buffer.template copy_from(mirror);
        };

    };

}