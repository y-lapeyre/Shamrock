#include "CommImplBuffer.hpp"
#include "shamsys/legacy/log.hpp"


namespace shamsys::comm::details {


    template<class T> 
    class CommBuffer<sycl::buffer<T>,CopyToHost>{
        
        T* usm_ptr;
        CommDetails<sycl::buffer<T>> details;
        
        public:
        CommBuffer(CommDetails<sycl::buffer<T>> det) : details(det){
            if(!det.comm_len){
                throw std::invalid_argument("cannot construct a buffer with a detail that doesn't specify the lenght");
            }
            usm_ptr = sycl::malloc_host<T>(det.comm_len,instance::get_compute_queue());
        }
        CommBuffer( sycl::buffer<T> & obj_ref){

            u64 len = obj_ref.size();

            details = CommDetails<sycl::buffer<T>>{len,{}};

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

            details = CommDetails<sycl::buffer<T>>{len,{}};

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
            details(other.details) {} // move constructor

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



            //{
            //    sycl::host_accessor acc {moved_obj};
            //    for(u64 sz = 0;sz < len; sz ++){
            //        usm_ptr[sz] = acc[sz + off];
            //    }
            //}

        }

        //void copy_back(sycl::buffer<T> & buf){
        //    
        //}

        static sycl::buffer<T> convert(CommBuffer && buf){

        }

        CommRequest<sycl::buffer<T>, DirectGPU> isend(u32 rank_dest, u32 comm_flag, MPI_Comm comm){

        }
        CommRequest<sycl::buffer<T>, DirectGPU> irecv(u32 rank_src, u32 comm_flag, MPI_Comm comm){

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

        CommRequest<sycl::buffer<T>, DirectGPU> isend(u32 rank_dest, u32 comm_flag, MPI_Comm comm){

        }
        CommRequest<sycl::buffer<T>, DirectGPU> irecv(u32 rank_src, u32 comm_flag, MPI_Comm comm){

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

        CommRequest<sycl::buffer<T>, DirectGPU> isend(u32 rank_dest, u32 comm_flag, MPI_Comm comm){

        }
        CommRequest<sycl::buffer<T>, DirectGPU> irecv(u32 rank_src, u32 comm_flag, MPI_Comm comm){

        }
    };




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