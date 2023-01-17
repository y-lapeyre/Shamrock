// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamsys/CommProtocol.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/SyclHelper.hpp"
#include "shamsys/SyclMpiTypes.hpp"

#include <optional>

namespace shamsys::comm::details {


    template<class T> 
    class CommDetails<sycl::buffer<T>>{public:
        u64 comm_len;
        std::optional<u64> start_index;
    };

    template<class T, Protocol comm_mode> 
    class CommBuffer<sycl::buffer<T>,comm_mode>{
        
        T* usm_ptr;
        CommDetails<sycl::buffer<T>> details;
        
        public:
        CommBuffer(CommDetails<sycl::buffer<T>> det);
        CommBuffer( sycl::buffer<T> & obj_ref);
        CommBuffer( sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det);
        CommBuffer( sycl::buffer<T> && moved_obj);
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det);


        ~CommBuffer();
        CommBuffer(CommBuffer&& other) noexcept; // move constructor
        CommBuffer& operator=(CommBuffer&& other) noexcept; // move assignment
        CommBuffer(const CommBuffer& other) =delete ;// copy constructor
        CommBuffer& operator=(const CommBuffer& other) = delete; // copy assignment




        sycl::buffer<T> copy_back();
        //void copy_back(sycl::buffer<T> & dest);
        static sycl::buffer<T> convert(CommBuffer && buf);

        void isend(CommRequests & rqs, u32 rank_dest, u32 comm_flag, MPI_Comm comm);
        void irecv(CommRequests & rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm);
    };








    template<class T> 
    class CommBuffer<sycl::buffer<T>,CopyToHost>{
        
        T* usm_ptr;
        CommDetails<sycl::buffer<T>> details;

        void alloc_usm(u64 len);
        void copy_to_usm(sycl::buffer<T> & obj_ref, u64 len, u64 offset);
        sycl::buffer<T> build_from_usm(u64 len, u64 offset);

        public:

        CommBuffer(CommDetails<sycl::buffer<T>> det) : details(det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
            alloc_usm(det.comm_len);
        }

        CommBuffer( sycl::buffer<T> & obj_ref){

            u64 len = obj_ref.size();

            details.comm_len = len;
            details.start_index = {};

            alloc_usm(len);
            copy_to_usm(obj_ref,len,0);

        }
        CommBuffer( sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det) : details(det){

            u64 len, off;
            len = det.comm_len;

            if(det.start_index){
                off = *det.start_index;
            }else{
                off = 0;
            }
            
            if(len + off > obj_ref.size()){
                throw std::invalid_argument("the offset + size request will create an overflow");
            }

            alloc_usm(len);
            copy_to_usm(obj_ref,len,off);

        }
        CommBuffer( sycl::buffer<T> && moved_obj){
            u64 len = moved_obj.size();

            details.comm_len = len;
            details.start_index = {};

            alloc_usm(len);
            copy_to_usm(moved_obj,len,0);
        }
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det) : details(det){

            u64 len, off;
            len = det.comm_len;

            if(det.start_index){
                off = *det.start_index;
            }else{
                off = 0;
            }
            
            if(len + off > moved_obj.size()){
                throw std::invalid_argument("the offset + size request will create an overflow");
            }

            alloc_usm(len);
            copy_to_usm(moved_obj,len,off);

        }

        ~CommBuffer(){
            //logger::raw_ln("~CommBuffer()");
            sycl::free(usm_ptr,instance::get_compute_queue());
        }


        CommBuffer(CommBuffer&& other) noexcept : 
            usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
            details(std::move(other.details)){

        } // move constructor

        CommBuffer& operator=(CommBuffer&& other) noexcept{
            std::swap(usm_ptr, other.usm_ptr);
            details = std::move(other.details);

            return *this;
        } // move assignment

        sycl::buffer<T> copy_back(){
            u64 len, off;
            len = details.comm_len;

            if(details.start_index){
                off = *details.start_index;
            }else{
                off = 0;
            }

            return build_from_usm(len,off);
        }

        //void copy_back(sycl::buffer<T> & buf){
        //    
        //}

        static sycl::buffer<T> convert(CommBuffer && buf){
            return buf.copy_back();
        }

        void isend(CommRequests & rqs, u32 rank_dest, u32 comm_tag, MPI_Comm comm);

        void irecv(CommRequests & rqs, u32 rank_src, u32 comm_tag, MPI_Comm comm);

    };



    template<class T> 
    class CommBuffer<sycl::buffer<T>,DirectGPU>{
        
        T* usm_ptr;
        CommDetails<sycl::buffer<T>> details;

        void alloc_usm(u64 len);

        void copy_to_usm(sycl::buffer<T> & obj_ref, u64 len, u64 offset);

        sycl::buffer<T> build_from_usm(u64 len, u64 offset);

        public:

        CommBuffer(CommDetails<sycl::buffer<T>> det) : details(det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
            alloc_usm(det.comm_len);
        }

        CommBuffer( sycl::buffer<T> & obj_ref){

            u64 len = obj_ref.size();

            details.comm_len = len;
            details.start_index = {};

            alloc_usm(len);
            copy_to_usm(obj_ref,len,0);

        }
        CommBuffer( sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det) : details(det){

            u64 len, off;
            len = det.comm_len;

            if(det.start_index){
                off = *det.start_index;
            }else{
                off = 0;
            }
            
            if(len + off > obj_ref.size()){
                throw std::invalid_argument("the offset + size request will create an overflow");
            }

            alloc_usm(len);
            copy_to_usm(obj_ref,len,off);

        }
        CommBuffer( sycl::buffer<T> && moved_obj){
            u64 len = moved_obj.size();

            details.comm_len = len;
            details.start_index = {};

            alloc_usm(len);
            copy_to_usm(moved_obj,len,0);
        }
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det) : details(det){

            u64 len, off;
            len = det.comm_len;

            if(det.start_index){
                off = *det.start_index;
            }else{
                off = 0;
            }
            
            if(len + off > moved_obj.size()){
                throw std::invalid_argument("the offset + size request will create an overflow");
            }

            alloc_usm(len);
            copy_to_usm(moved_obj,len,off);

        }

        ~CommBuffer(){
            //logger::raw_ln("~CommBuffer()");
            sycl::free(usm_ptr,instance::get_compute_queue());
        }


        CommBuffer(CommBuffer&& other) noexcept : 
            usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
            details(std::move(other.details)){

        } // move constructor

        CommBuffer& operator=(CommBuffer&& other) noexcept{
            std::swap(usm_ptr, other.usm_ptr);
            details = std::move(other.details);

            return *this;
        } // move assignment

        sycl::buffer<T> copy_back(){
            u64 len, off;
            len = details.comm_len;

            if(details.start_index){
                off = *details.start_index;
            }else{
                off = 0;
            }

            return build_from_usm(len,off);
        }

        //void copy_back(sycl::buffer<T> & buf){
        //    
        //}

        static sycl::buffer<T> convert(CommBuffer && buf){
            return buf.copy_back();
        }

        void isend(CommRequests & rqs, u32 rank_dest, u32 comm_tag, MPI_Comm comm);

        void irecv(CommRequests & rqs, u32 rank_src, u32 comm_tag, MPI_Comm comm);

    };
































    template<class T> 
    class CommBuffer<sycl::buffer<T>,DirectGPUFlatten>{

        using ptr_t = typename shamsys::syclhelper::get_base_sycl_type<T>::type;
        constexpr static u64 int_len = shamsys::syclhelper::get_base_sycl_type<T>::int_len;

        ptr_t* usm_ptr;
        CommDetails<sycl::buffer<T>> details;

        void alloc_usm(u64 len);
        void copy_to_usm(sycl::buffer<T> & obj_ref, u64 len, u64 offset);
        sycl::buffer<T> build_from_usm(u64 len, u64 offset);

        public:

        CommBuffer(CommDetails<sycl::buffer<T>> det) : details(det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
            alloc_usm(det.comm_len);
        }

        CommBuffer( sycl::buffer<T> & obj_ref){

            u64 len = obj_ref.size();

            details.comm_len = len;
            details.start_index = {};

            alloc_usm(len);
            copy_to_usm(obj_ref,len,0);

        }
        CommBuffer( sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det) : details(det){

            u64 len, off;
            len = det.comm_len;

            if(det.start_index){
                off = *det.start_index;
            }else{
                off = 0;
            }
            
            if(len + off > obj_ref.size()){
                throw std::invalid_argument("the offset + size request will create an overflow");
            }

            alloc_usm(len);
            copy_to_usm(obj_ref,len,off);

        }
        CommBuffer( sycl::buffer<T> && moved_obj){
            u64 len = moved_obj.size();

            details.comm_len = len;
            details.start_index = {};

            alloc_usm(len);
            copy_to_usm(moved_obj,len,0);
        }
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det) : details(det){

            u64 len, off;
            len = det.comm_len;

            if(det.start_index){
                off = *det.start_index;
            }else{
                off = 0;
            }
            
            if(len + off > moved_obj.size()){
                throw std::invalid_argument("the offset + size request will create an overflow");
            }

            alloc_usm(len);
            copy_to_usm(moved_obj,len,off);

        }

        ~CommBuffer(){
            //logger::raw_ln("~CommBuffer()");
            sycl::free(usm_ptr,instance::get_compute_queue());
        }


        CommBuffer(CommBuffer&& other) noexcept : 
            usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
            details(std::move(other.details)){

        } // move constructor

        CommBuffer& operator=(CommBuffer&& other) noexcept{
            std::swap(usm_ptr, other.usm_ptr);
            details = std::move(other.details);

            return *this;
        } // move assignment

        sycl::buffer<T> copy_back(){
            u64 len, off;
            len = details.comm_len;

            if(details.start_index){
                off = *details.start_index;
            }else{
                off = 0;
            }

            return build_from_usm(len,off);
        }

        //void copy_back(sycl::buffer<T> & buf){
        //    
        //}

        static sycl::buffer<T> convert(CommBuffer && buf){
            return buf.copy_back();
        }

        void isend(CommRequests & rqs, u32 rank_dest, u32 comm_tag, MPI_Comm comm);

        void irecv(CommRequests & rqs, u32 rank_src, u32 comm_tag, MPI_Comm comm);

    };





    
} // namespace shamsys::comm::details