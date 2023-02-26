// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#pragma once

#include "shamsys/MpiWrapper.hpp"
#include "shamsys/comm/ProtocolEnum.hpp"
#include "shamsys/legacy/log.hpp"

#include <stdexcept>
#include <variant>
#include <vector>


#include "CommRequests.hpp"
#include "shamutils/throwUtils.hpp"
//#include "CommImplBuffer.hpp"



namespace shamsys::comm::details {

    template<class T> class CommDetails;
    template<class T, Protocol comm_mode> class CommBuffer;
    
} // namespace shamsys::comm::details


namespace shamsys::comm {

    ///// forward declaration /////
    template<class T> using CommDetails = details::CommDetails<T>;
    template<class T> class CommBuffer;


    


    ///// class implementations /////

    template<class T> class CommBuffer{
        private:

        using var_t = 
            std::optional<
                std::variant<
                    details::CommBuffer<T,CopyToHost>      ,
                    details::CommBuffer<T,DirectGPU>       ,
                    details::CommBuffer<T,DirectGPUFlatten>
                >
            >;

        var_t _int_type;


        explicit CommBuffer (var_t && moved_int_var) : _int_type(std::move(moved_int_var)) {}

        public:

        /**
         * @brief Construct a CommBuffer with sizes according to det, and with comm_mode as protocol
         * 
         * @param det 
         * @param comm_mode 
         */
        CommBuffer(CommDetails<T> det, Protocol comm_mode) {
            if(comm_mode == CopyToHost){
                _int_type = details::CommBuffer<T, CopyToHost>(det);
            }else if(comm_mode == DirectGPU){
                _int_type = details::CommBuffer<T, DirectGPU>(det);
            }else if(comm_mode == DirectGPUFlatten){
                _int_type = details::CommBuffer<T, DirectGPUFlatten>(det);
            }else {
                throw shamutils::throw_with_loc<std::invalid_argument>("unknown mode");
            }
        }

        /**
         * @brief Construct a CommBuffer containing a copy of the content of obj_ref
         * 
         * @param obj_ref 
         * @param comm_mode 
         */
        CommBuffer( T & obj_ref, Protocol comm_mode) {
            if(comm_mode == CopyToHost){
                _int_type = details::CommBuffer<T, CopyToHost>(obj_ref);
            }else if(comm_mode == DirectGPU){
                _int_type = details::CommBuffer<T, DirectGPU>(obj_ref);
            }else if(comm_mode == DirectGPUFlatten){
                _int_type = details::CommBuffer<T, DirectGPUFlatten>(obj_ref);
            }else {
                throw shamutils::throw_with_loc<std::invalid_argument>("unknown mode");
            }
        }

        /**
         * @brief  Construct a CommBuffer containing a copy of the content of obj_ref built from the details in det
         * 
         * @param obj_ref 
         * @param det 
         * @param comm_mode 
         */
        CommBuffer( T & obj_ref, CommDetails<T> det, Protocol comm_mode) {
            if(comm_mode == CopyToHost){
                _int_type = details::CommBuffer<T, CopyToHost>(obj_ref,det);
            }else if(comm_mode == DirectGPU){
                _int_type = details::CommBuffer<T, DirectGPU>(obj_ref,det);
            }else if(comm_mode == DirectGPUFlatten){
                _int_type = details::CommBuffer<T, DirectGPUFlatten>(obj_ref,det);
            }else {
                throw shamutils::throw_with_loc<std::invalid_argument>("unknown mode");
            }
        }

        /**
         * @brief Construct a CommBuffer containing the moved object "moved_obj"
         * 
         * @param moved_obj 
         * @param comm_mode 
         */
        CommBuffer( T && moved_obj, Protocol comm_mode) {
            if(comm_mode == CopyToHost){
                _int_type = details::CommBuffer<T, CopyToHost>(moved_obj);
            }else if(comm_mode == DirectGPU){
                _int_type = details::CommBuffer<T, DirectGPU>(moved_obj);
            }else if(comm_mode == DirectGPUFlatten){
                _int_type = details::CommBuffer<T, DirectGPUFlatten>(moved_obj);
            }else {
                throw shamutils::throw_with_loc<std::invalid_argument>("unknown mode");
            }
        }

        /**
         * @brief  Construct a CommBuffer containing the moved object "moved_obj" built from the details in det
         * 
         * @param moved_obj 
         * @param det 
         * @param comm_mode 
         */
        CommBuffer( T && moved_obj, CommDetails<T> det, Protocol comm_mode) {
            if(comm_mode == CopyToHost){
                _int_type = details::CommBuffer<T, CopyToHost>(moved_obj,det);
            }else if(comm_mode == DirectGPU){
                _int_type = details::CommBuffer<T, DirectGPU>(moved_obj,det);
            }else if(comm_mode == DirectGPUFlatten){
                _int_type = details::CommBuffer<T, DirectGPUFlatten>(moved_obj,det);
            }else {
                throw shamutils::throw_with_loc<std::invalid_argument>("unknown mode");
            }
        }



        /// obj recovery funcs


        /**
         * @brief return a copy of the held object in the buffer
         * 
         * @return T 
         */
        T copy_back(){
            return std::visit([=](auto && arg) {
                return arg.copy_back();
            }, *_int_type);
        }

        ///**
        // * @brief return a copy of the held object in the buffer
        // * 
        // * @param des destination of the copy
        // */
        //void copy_back(T & dest){
        //    std::visit([&](auto && arg) {
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
            }, *buf._int_type);
        }


        /// implement comm functions

        /**
         * @brief similar to MPI_ISend but with the protocols containers
         * 
         * @param rqs 
         * @param rank_dest 
         * @param comm_tag 
         * @param comm 
         */
        void isend(CommRequests & rqs, u32 rank_dest, u32 comm_tag, MPI_Comm comm){
            std::visit([&](auto&& arg) {
                arg.isend(rqs, rank_dest, comm_tag, comm);
            }, *_int_type);
        }

        /**
         * @brief similar to MPI_IRecv but with the protocols containers
         * 
         * @param rqs 
         * @param rank_src 
         * @param comm_tag 
         * @param comm 
         */
        void irecv(CommRequests & rqs, u32 rank_src, u32 comm_tag, MPI_Comm comm){
            std::visit([&](auto&& arg) {
                arg.irecv(rqs, rank_src, comm_tag, comm);
            }, *_int_type);
        }


        static CommBuffer irecv_probe(CommRequests & rqs,u32 rank_src, u32 comm_flag, MPI_Comm comm, Protocol comm_mode, CommDetails<T> details){

            if(comm_mode == CopyToHost){
                return CommBuffer(details::CommBuffer<T, CopyToHost>::irecv_probe(rqs, rank_src,comm_flag,comm,details));
            }
            
            if(comm_mode == DirectGPU){
                return CommBuffer(details::CommBuffer<T, DirectGPU>::irecv_probe(rqs, rank_src,comm_flag,comm,details));
            }
            
            if(comm_mode == DirectGPUFlatten){
                return CommBuffer(details::CommBuffer<T, DirectGPUFlatten>::irecv_probe(rqs, rank_src,comm_flag,comm,details));
            }

            throw shamutils::throw_with_loc<std::invalid_argument>("unknown mode");

        }

    };

} // namespace shamsys::comm

