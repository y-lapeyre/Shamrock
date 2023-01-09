#include "CommImplBuffer.hpp"


namespace shamsys::comm::details {


    template<class T> 
    class CommBuffer<sycl::buffer<T>,CopyToHost>{
        
        T* usm_ptr;
        
        public:
        CommBuffer(CommDetails<sycl::buffer<T>> det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
        }
        CommBuffer( const sycl::buffer<T> & obj_ref){

        }
        CommBuffer( const sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det){

        }
        CommBuffer( sycl::buffer<T> && moved_obj){

        }
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det){

        }

        ~CommBuffer(){
            sycl::free(usm_ptr,instance::get_compute_queue());
        }

        T copy_back(){

        }
        static T convert(CommBuffer && buf){

        }

        CommRequest<sycl::buffer<T>, DirectGPU> isend(u32 rank_dest, u32 comm_flag, MPI_Comm comm){

        }
        CommRequest<sycl::buffer<T>, DirectGPU> irecv(u32 rank_src, u32 comm_flag, MPI_Comm comm){

        }

    };


    template<class T> 
    class CommBuffer<sycl::buffer<T>,DirectGPU>{
        
        T* usm_ptr;
        
        public:
        CommBuffer(CommDetails<sycl::buffer<T>> det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
        }
        CommBuffer( const sycl::buffer<T> & obj_ref){

        }
        CommBuffer( const sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det){

        }
        CommBuffer( sycl::buffer<T> && moved_obj){

        }
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det){

        }

        ~CommBuffer(){
            sycl::free(usm_ptr,instance::get_compute_queue());
        }

        T copy_back(){

        }
        static T convert(CommBuffer && buf){

        }

        CommRequest<sycl::buffer<T>, DirectGPU> isend(u32 rank_dest, u32 comm_flag, MPI_Comm comm){

        }
        CommRequest<sycl::buffer<T>, DirectGPU> irecv(u32 rank_src, u32 comm_flag, MPI_Comm comm){

        }
    };

    template<class T> 
    class CommBuffer<sycl::buffer<T>,DirectGPUFlatten>{
        
        T* usm_ptr;
        
        public:
        CommBuffer(CommDetails<sycl::buffer<T>> det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
        }
        CommBuffer( const sycl::buffer<T> & obj_ref){

        }
        CommBuffer( const sycl::buffer<T> & obj_ref, CommDetails<sycl::buffer<T>> det){

        }
        CommBuffer( sycl::buffer<T> && moved_obj){

        }
        CommBuffer( sycl::buffer<T> && moved_obj, CommDetails<sycl::buffer<T>> det){

        }
        ~CommBuffer(){
            sycl::free(usm_ptr,instance::get_compute_queue());
        }

        T copy_back(){

        }
        static T convert(CommBuffer && buf){

        }

        CommRequest<sycl::buffer<T>, DirectGPU> isend(u32 rank_dest, u32 comm_flag, MPI_Comm comm){

        }
        CommRequest<sycl::buffer<T>, DirectGPU> irecv(u32 rank_src, u32 comm_flag, MPI_Comm comm){

        }
    };




    //template<class T, Protocol comm_mode> 
    //class CommRequest<sycl::buffer<T>,comm_mode>{
//
    //};

    template class CommBuffer<sycl::buffer<f32_3>,CopyToHost>;
    template class CommBuffer<sycl::buffer<f32_3>,DirectGPU>;
    template class CommBuffer<sycl::buffer<f32_3>,DirectGPUFlatten>;
    
} // namespace shamsys::comm::details