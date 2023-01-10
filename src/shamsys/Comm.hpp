/*
cf https://github.com/tdavidcl/Shamrock/issues/23 


we want to user side to look like this


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

Communicator<... type ...> comm {MPI_COMM_WORLD, protocol::DirectGPU};

auto tmp = comm.prepare_send_full(... obj to send ...);
auto tmp = comm.prepare_send(... obj to send ...,details<... type ...>{.....});

// ... do comm calls
comm.isend(tmp,0 ,0);
// ...

comm.sync(); // note : sync only sync sycl with MPI, ie can be nonblocking

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

*/

#pragma once

#include "MpiWrapper.hpp"
#include "legacy/log.hpp"

#include <variant>
#include <vector>


#include "CommProtocol.hpp"
#include "CommImplBuffer.hpp"





namespace shamsys::comm {

    ///// forward declaration /////
    template<class T> using CommDetails = details::CommDetails<T>;
    template<class T> class CommBuffer;
    template<class T> class CommRequest;


    ///// class implementations /////

    template<class T> class CommBuffer{
        private:

        using var_t = std::variant<
            details::CommBuffer<T,CopyToHost>,
            details::CommBuffer<T,DirectGPU>,
            details::CommBuffer<T,DirectGPUFlatten>
        >;

        var_t _int_type;


        static constexpr auto build_variant = [](Protocol comm_mode, auto ... args) -> var_t{
            if(comm_mode == CopyToHost){
                return details::CommBuffer<T, CopyToHost>(args...);
            }else if(comm_mode == DirectGPU){
                return details::CommBuffer<T, DirectGPU>(args...);
            }else if(comm_mode == DirectGPUFlatten){
                return details::CommBuffer<T, DirectGPUFlatten>(args...);
            }else{
                throw std::invalid_argument("Protocol value selected does not fit within listed cases");
            }
        };

        public:

        /**
         * @brief Construct a CommBuffer with sizes according to det, and with comm_mode as protocol
         * 
         * @param det 
         * @param comm_mode 
         */
        CommBuffer(CommDetails<T> det, Protocol comm_mode) : _int_type(build_variant(comm_mode,det)){}

        /**
         * @brief Construct a CommBuffer containing a copy of the content of obj_ref
         * 
         * @param obj_ref 
         * @param comm_mode 
         */
        CommBuffer( T & obj_ref, Protocol comm_mode) : _int_type(build_variant(comm_mode,obj_ref)){}

        /**
         * @brief  Construct a CommBuffer containing a copy of the content of obj_ref built from the details in det
         * 
         * @param obj_ref 
         * @param det 
         * @param comm_mode 
         */
        CommBuffer( T & obj_ref, CommDetails<T> det, Protocol comm_mode) : _int_type(build_variant(comm_mode,obj_ref,det)){}

        /**
         * @brief Construct a CommBuffer containing the moved object "moved_obj"
         * 
         * @param moved_obj 
         * @param comm_mode 
         */
        CommBuffer( T && moved_obj, Protocol comm_mode) : _int_type(build_variant(comm_mode,moved_obj)){}

        /**
         * @brief  Construct a CommBuffer containing the moved object "moved_obj" built from the details in det
         * 
         * @param moved_obj 
         * @param det 
         * @param comm_mode 
         */
        CommBuffer( T && moved_obj, CommDetails<T> det, Protocol comm_mode) : _int_type(build_variant(comm_mode,moved_obj,det)){}


        /// obj recovery funcs


        /**
         * @brief return a copy of the held object in the buffer
         * 
         * @return T 
         */
        T copy_back(){
            return std::visit([=](auto&& arg) {
                return arg.copy_back();
            }, _int_type);
        }

        ///**
        // * @brief return a copy of the held object in the buffer
        // * 
        // * @param des destination of the copy
        // */
        //void copy_back(T & dest){
        //    std::visit([=](auto & arg) {
        //        arg.copy_back(dest);
        //    }, _int_type);
        //}

        /**
         * @brief destroy the buffer and recover the held object
         * 
         * @param buf 
         * @return T 
         */
        static T convert(CommBuffer && buf){
            return std::visit([=](auto&& arg) {
                using _t = typename std::remove_reference<decltype(arg)>::type;
                return _t::convert(std::forward<_t>(arg));
            }, buf._int_type);
        }


        /// implement comm functions

        CommRequest<T> isend(u32 rank_dest, u32 comm_flag, MPI_Comm comm){
            return std::visit([=](auto&& arg) {
                return CommRequest<T>(arg.isend(rank_dest, comm_flag, comm));
            }, _int_type);
        }

        CommRequest<T> irecv(u32 rank_src, u32 comm_flag, MPI_Comm comm){
            return std::visit([=](auto&& arg) {
                return CommRequest<T>(arg.irecv(rank_src, comm_flag, comm));
            }, _int_type);
        }

    };

    template<class T> class CommRequest{
        private:
        std::variant<
            details::CommRequest<T,CopyToHost>,
            details::CommRequest<T,DirectGPU>,
            details::CommRequest<T,DirectGPUFlatten>
        > _int_type;

        public:

        template<Protocol comm_mode> explicit CommRequest(details::CommRequest<T,comm_mode> rq) : _int_type(rq){}
    };


    template class CommBuffer<sycl::buffer<f32_3>>;

} // namespace shamsys::comm

