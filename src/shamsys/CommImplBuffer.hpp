#pragma once

#include "shamsys/CommProtocol.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/MpiWrapper.hpp"

#include <optional>

namespace shamsys::comm::details {

    template<class T> 
    class CommDetails<sycl::buffer<T>>{public:
        std::optional<u64> start_index;
        std::optional<u64> comm_len;
    };

    template<class T, Protocol comm_mode> 
    class CommBuffer<sycl::buffer<T>,comm_mode>{public:
        CommBuffer(CommDetails<sycl::buffer<T>> det);
        CommBuffer( const sycl::buffer<T> & obj_ref);
        CommBuffer( const sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det);
        CommBuffer( sycl::buffer<T> && moved_obj);
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det);
        ~CommBuffer();


        T copy_back();
        static T convert(CommBuffer && buf);

        CommRequest<sycl::buffer<T>, comm_mode> isend(u32 rank_dest, u32 comm_flag, MPI_Comm comm);
        CommRequest<sycl::buffer<T>, comm_mode> irecv(u32 rank_src, u32 comm_flag, MPI_Comm comm);
    };

    template<class T, Protocol comm_mode> 
    class CommRequest<sycl::buffer<T>,comm_mode>{

    };
    
} // namespace shamsys::comm::details