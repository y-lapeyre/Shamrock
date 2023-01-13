#pragma once

#include "shamsys/CommProtocol.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/SyclMpiTypes.hpp"

#include <optional>

namespace shamsys::comm::details {

    using CommRequests = std::vector<MPI_Request>;

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

        CommRequest<sycl::buffer<T>, comm_mode> isend(u32 rank_dest, u32 comm_tag, MPI_Comm comm);
        CommRequest<sycl::buffer<T>, comm_mode> irecv(u32 rank_src, u32 comm_tag, MPI_Comm comm);
    };




    template<class T>
    class CommRequest<sycl::buffer<T>,CopyToHost>{
        

        public:
        void wait(){

        }
    };




    template<class T> 
    class CommBuffer<sycl::buffer<T>,CopyToHost>{
        
        T* usm_ptr;
        CommDetails<sycl::buffer<T>> details;

        using Rq_t = CommRequest<sycl::buffer<T>, CopyToHost>;
        
        public:

        CommBuffer(CommDetails<sycl::buffer<T>> det) : details(det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
            usm_ptr = sycl::malloc_host<T>(det.comm_len,instance::get_compute_queue());
        }

        CommBuffer( sycl::buffer<T> & obj_ref){

            u64 len = obj_ref.size();

            details.comm_len = len;
            details.start_index = {};

            usm_ptr = sycl::malloc_host<T>(len,instance::get_compute_queue());

            {
                sycl::host_accessor acc {obj_ref};
                for(u64 sz = 0;sz < len; sz ++){
                    usm_ptr[sz] = acc[sz];
                }
            }

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

            usm_ptr = sycl::malloc_host<T>(len,instance::get_compute_queue());

            {
                sycl::host_accessor acc {obj_ref};
                for(u64 sz = 0;sz < len; sz ++){
                    usm_ptr[sz] = acc[sz + off];
                }
            }

        }
        CommBuffer( sycl::buffer<T> && moved_obj){
            u64 len = moved_obj.size();

            details.comm_len = len;
            details.start_index = {};

            usm_ptr = sycl::malloc_host<T>(len,instance::get_compute_queue());

            {
                sycl::host_accessor acc {moved_obj};
                for(u64 sz = 0;sz < len; sz ++){
                    usm_ptr[sz] = acc[sz];
                }
            }
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

            usm_ptr = sycl::malloc_host<T>(len,instance::get_compute_queue());

            {
                sycl::host_accessor acc {moved_obj};
                for(u64 sz = 0;sz < len; sz ++){
                    usm_ptr[sz] = acc[sz + off];
                }
            }

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

            sycl::buffer<T> buf_ret (details.comm_len);

            {
                sycl::host_accessor acc {buf_ret};
                for(u64 sz = 0;sz < len; sz ++){
                    acc[sz + off] = usm_ptr[sz];
                }
            }

            return buf_ret;
        }

        //void copy_back(sycl::buffer<T> & buf){
        //    
        //}

        static sycl::buffer<T> convert(CommBuffer && buf){
            return buf.copy_back();
        }

        void isend(CommRequests & rqs, u32 rank_dest, u32 comm_tag, MPI_Comm comm){
            MPI_Request rq;
            mpi::isend(
                usm_ptr, 
                details.comm_len, 
                get_mpi_type<T>(), 
                rank_dest, 
                comm_tag, 
                comm, 
                rq);
            rqs.push_back(rq);
        }

        void irecv(CommRequests & rqs, u32 rank_src, u32 comm_tag, MPI_Comm comm){
            MPI_Request rq;
            mpi::irecv(usm_ptr, details.comm_len, get_mpi_type<T>(), rank_src, comm_tag, comm, rq);
        }

    };





    template<class T> 
    class CommBuffer<sycl::buffer<T>,DirectGPU>{
        
        T* usm_ptr;
        CommDetails<sycl::buffer<T>> details;
        
        public:
        CommBuffer(CommDetails<sycl::buffer<T>> det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
        }
        CommBuffer( sycl::buffer<T> & obj_ref){

        }
        CommBuffer( sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det){

        }
        CommBuffer( sycl::buffer<T> && moved_obj){

        }
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det){

        }

        ~CommBuffer(){
            sycl::free(usm_ptr,instance::get_compute_queue());
        }

        CommBuffer(CommBuffer&& other) noexcept : 
            usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
            details(other.details) {} // move constructor

        CommBuffer& operator=(CommBuffer&& other) noexcept{
            std::swap(usm_ptr, other.usm_ptr);
            details = std::move(other.details);
            return *this;
        } // move assignment

        sycl::buffer<T> copy_back(){

        }
        static sycl::buffer<T> convert(CommBuffer && buf){

        }

        void isend(CommRequests & rqs, u32 rank_dest, u32 comm_flag, MPI_Comm comm){

        }
        void irecv(CommRequests & rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm){

        }
    };

    template<class T> 
    class CommBuffer<sycl::buffer<T>,DirectGPUFlatten>{
        
        T* usm_ptr;
        CommDetails<sycl::buffer<T>> details;
        
        public:
        CommBuffer(CommDetails<sycl::buffer<T>> det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
        }
        CommBuffer( sycl::buffer<T> & obj_ref){

        }
        CommBuffer( sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det){

        }
        CommBuffer( sycl::buffer<T> && moved_obj){

        }
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det){

        }
        ~CommBuffer(){
            sycl::free(usm_ptr,instance::get_compute_queue());
        }

        CommBuffer(CommBuffer&& other) noexcept : 
            usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
            details(other.details) {} // move constructor

        CommBuffer& operator=(CommBuffer&& other) noexcept{
            std::swap(usm_ptr, other.usm_ptr);
            details = std::move(other.details);
            return *this;
        } // move assignment

        sycl::buffer<T> copy_back(){

        }
        static sycl::buffer<T> convert(CommBuffer && buf){

        }

        void isend(CommRequests & rqs, u32 rank_dest, u32 comm_flag, MPI_Comm comm){

        }
        void irecv(CommRequests & rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm){

        }
    };







    template<class T, Protocol comm_mode> 
    class CommRequest<sycl::buffer<T>,comm_mode>{

    };
    
} // namespace shamsys::comm::details