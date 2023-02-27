// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "CommImplBuffer.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamutils/throwUtils.hpp"
#include <stdexcept>


namespace shamsys::comm::details {

    //////////////////////////////////////////
    //copy to host impl
    //////////////////////////////////////////

    template<class T>
    void CommBuffer<sycl::buffer<T>,CopyToHost>::alloc_usm(u64 len){
        usm_ptr = sycl::malloc_host<T>(len,instance::get_compute_queue());
    }

    template<class T>
    void CommBuffer<sycl::buffer<T>,CopyToHost>::copy_to_usm(sycl::buffer<T> & obj_ref, u64 len, u64 offset){
            
            sycl::host_accessor acc {obj_ref, sycl::read_only};
            for(u64 sz = 0;sz < len; sz ++){
                usm_ptr[sz] = acc[sz + offset];
            }
        
        }
    template<class T>
    sycl::buffer<T> CommBuffer<sycl::buffer<T>,CopyToHost>::build_from_usm(u64 len, u64 offset){

        sycl::buffer<T> buf_ret (len);
        {
            sycl::host_accessor acc {buf_ret, sycl::write_only, sycl::no_init};
            for(u64 sz = 0;sz < len; sz ++){
                acc[sz + offset] = usm_ptr[sz];
            }
        }
        return buf_ret;
    }

template<class T>
    void CommBuffer<sycl::buffer<T>,CopyToHost>::isend(CommRequests & rqs, u32 rank_dest, u32 comm_tag, MPI_Comm comm){
            MPI_Request rq;
            mpi::isend(
                usm_ptr, 
                details.comm_len, 
                get_mpi_type<T>(), 
                rank_dest, 
                comm_tag, 
                comm, 
                &rq);
            rqs.push(rq);
        }

template<class T>
        void CommBuffer<sycl::buffer<T>,CopyToHost>::irecv(CommRequests & rqs, u32 rank_src, u32 comm_tag, MPI_Comm comm){
            MPI_Request rq;
            mpi::irecv(usm_ptr, details.comm_len, get_mpi_type<T>(), rank_src, comm_tag, comm, &rq);
            rqs.push(rq);
        }

template<class T>
    CommBuffer<sycl::buffer<T>,CopyToHost> CommBuffer<sycl::buffer<T>,CopyToHost>::irecv_probe(CommRequests & rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm, CommDetails<sycl::buffer<T>> details){
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_src, comm_flag,comm, & st);
        mpi::get_count(&st, get_mpi_type<T>(), &cnt);

        u32 val_cnt = cnt;

        CommDetails<sycl::buffer<T>> det {val_cnt};

        CommBuffer<sycl::buffer<T>,CopyToHost> ret {det};

        ret.irecv(rqs, rank_src, comm_flag, comm);

        return ret;
    }



    //////////////////////////////////////////
    //direct GPU impl
    //////////////////////////////////////////

    template<class T>
    void CommBuffer<sycl::buffer<T>,DirectGPU>::alloc_usm(u64 len){
        usm_ptr = sycl::malloc_device<T>(len,instance::get_compute_queue());
    }

    template<class T>
    void CommBuffer<sycl::buffer<T>,DirectGPU>::copy_to_usm(sycl::buffer<T> & obj_ref, u64 len, u64 offset){
        
        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {obj_ref, cgh, sycl::read_only};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr[i] = acc_buf[i + off];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    
    }

    template<class T>
    sycl::buffer<T> CommBuffer<sycl::buffer<T>,DirectGPU>::build_from_usm(u64 len, u64 offset){

        sycl::buffer<T> buf_ret (len);


        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {buf_ret, cgh,sycl::write_only, sycl::no_init};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                acc_buf[i + off] = ptr[i];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls

        return buf_ret;
    }

    template<class T>
    void CommBuffer<sycl::buffer<T>,DirectGPU>::isend(CommRequests & rqs, u32 rank_dest, u32 comm_tag, MPI_Comm comm){
        MPI_Request rq;
        mpi::isend(
            usm_ptr, 
            details.comm_len, 
            get_mpi_type<T>(), 
            rank_dest, 
            comm_tag, 
            comm, 
            &rq);
        rqs.push(rq);
    }

    template<class T>
    void CommBuffer<sycl::buffer<T>,DirectGPU>::irecv(CommRequests & rqs, u32 rank_src, u32 comm_tag, MPI_Comm comm){
        MPI_Request rq;
        mpi::irecv(usm_ptr, details.comm_len, get_mpi_type<T>(), rank_src, comm_tag, comm, &rq);
        rqs.push(rq);
    }


template<class T>
    CommBuffer<sycl::buffer<T>,DirectGPU> CommBuffer<sycl::buffer<T>,DirectGPU>::irecv_probe(CommRequests & rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm, CommDetails<sycl::buffer<T>> details){
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_src, comm_flag,comm, & st);
        mpi::get_count(&st, get_mpi_type<T>(), &cnt);

        u32 val_cnt = cnt;

        CommDetails<sycl::buffer<T>> det {val_cnt};

        CommBuffer<sycl::buffer<T>,DirectGPU> ret {det};

        ret.irecv(rqs, rank_src, comm_flag, comm);

        return ret;
    }









    //////////////////////////////////////////
    //direct GPU flattened impl
    //////////////////////////////////////////

    template<class T>
    void CommBuffer<sycl::buffer<T>,DirectGPUFlatten>::alloc_usm(u64 len){
        usm_ptr = sycl::malloc_device<ptr_t>(len*int_len,instance::get_compute_queue());
    }



    

    template<class T>
    void flatten_copy_to_usm(sycl::buffer<T> & obj_ref, T* usm_ptr, u64 len, u64 offset){
        
        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {obj_ref, cgh, sycl::read_only};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr[i] = acc_buf[i + off];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    
    }

    template<class T>
    void flatten_build_from_usm(sycl::buffer<T> & buf_ret,T* usm_ptr, u64 len, u64 offset){

        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {buf_ret, cgh, sycl::write_only, sycl::no_init};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                acc_buf[i + off] = ptr[i];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls

    }


    template<class T>
    void flatten_copy_to_usm(sycl::buffer<sycl::vec<T,2>> & obj_ref, T* usm_ptr, u64 len, u64 offset){
        
        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {obj_ref, cgh, sycl::read_only};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr[i*2 + 0] = acc_buf[i + off].x();
                ptr[i*2 + 1] = acc_buf[i + off].y();
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    
    }

    template<class T>
    void flatten_build_from_usm(sycl::buffer<sycl::vec<T,2>> & buf_ret,T* usm_ptr, u64 len, u64 offset){

        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {buf_ret, cgh, sycl::write_only, sycl::no_init};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){

                acc_buf[i + off].x() = ptr[i*2 + 0];
                acc_buf[i + off].y() = ptr[i*2 + 1];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls

    }




    template<class T>
    void flatten_copy_to_usm(sycl::buffer<sycl::vec<T,3>> & obj_ref, T* usm_ptr, u64 len, u64 offset){
        
        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {obj_ref, cgh, sycl::read_only};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr[i*3 + 0] = acc_buf[i + off].x();
                ptr[i*3 + 1] = acc_buf[i + off].y();
                ptr[i*3 + 2] = acc_buf[i + off].z();
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    
    }

    template<class T>
    void flatten_build_from_usm(sycl::buffer<sycl::vec<T,3>> & buf_ret,T* usm_ptr, u64 len, u64 offset){

        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {buf_ret, cgh, sycl::write_only, sycl::no_init};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){

                acc_buf[i + off].x() = ptr[i*3 + 0];
                acc_buf[i + off].y() = ptr[i*3 + 1];
                acc_buf[i + off].z() = ptr[i*3 + 2];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls

    }

    template<class T>
    void flatten_copy_to_usm(sycl::buffer<sycl::vec<T,4>> & obj_ref, T* usm_ptr, u64 len, u64 offset){
        
        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {obj_ref, cgh, sycl::read_only};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr[i*4 + 0] = acc_buf[i + off].x();
                ptr[i*4 + 1] = acc_buf[i + off].y();
                ptr[i*4 + 2] = acc_buf[i + off].z();
                ptr[i*4 + 3] = acc_buf[i + off].w();
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    
    }

    template<class T>
    void flatten_build_from_usm(sycl::buffer<sycl::vec<T,4>> & buf_ret,T* usm_ptr, u64 len, u64 offset){

        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {buf_ret, cgh, sycl::write_only, sycl::no_init};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){

                acc_buf[i + off].x() = ptr[i*4 + 0];
                acc_buf[i + off].y() = ptr[i*4 + 1];
                acc_buf[i + off].z() = ptr[i*4 + 2];
                acc_buf[i + off].w() = ptr[i*4 + 3];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls

    }



    template<class T>
    void flatten_copy_to_usm(sycl::buffer<sycl::vec<T,8>> & obj_ref, T* usm_ptr, u64 len, u64 offset){
        
        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {obj_ref, cgh, sycl::read_only};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr[i*8 + 0] = acc_buf[i + off].s0();
                ptr[i*8 + 1] = acc_buf[i + off].s1();
                ptr[i*8 + 2] = acc_buf[i + off].s2();
                ptr[i*8 + 3] = acc_buf[i + off].s3();
                ptr[i*8 + 4] = acc_buf[i + off].s4();
                ptr[i*8 + 5] = acc_buf[i + off].s5();
                ptr[i*8 + 6] = acc_buf[i + off].s6();
                ptr[i*8 + 7] = acc_buf[i + off].s7();
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    
    }

    template<class T>
    void flatten_build_from_usm(sycl::buffer<sycl::vec<T,8>> & buf_ret,T* usm_ptr, u64 len, u64 offset){

        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {buf_ret, cgh, sycl::write_only, sycl::no_init};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){

                acc_buf[i + off].s0() = ptr[i*8 + 0];
                acc_buf[i + off].s1() = ptr[i*8 + 1];
                acc_buf[i + off].s2() = ptr[i*8 + 2];
                acc_buf[i + off].s3() = ptr[i*8 + 3];
                acc_buf[i + off].s4() = ptr[i*8 + 4];
                acc_buf[i + off].s5() = ptr[i*8 + 5];
                acc_buf[i + off].s6() = ptr[i*8 + 6];
                acc_buf[i + off].s7() = ptr[i*8 + 7];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls

    }



    template<class T>
    void flatten_copy_to_usm(sycl::buffer<sycl::vec<T,16>> & obj_ref, T* usm_ptr, u64 len, u64 offset){
        
        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {obj_ref, cgh, sycl::read_only};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr[i*16 + 0] = acc_buf[i + off].s0();
                ptr[i*16 + 1] = acc_buf[i + off].s1();
                ptr[i*16 + 2] = acc_buf[i + off].s2();
                ptr[i*16 + 3] = acc_buf[i + off].s3();
                ptr[i*16 + 4] = acc_buf[i + off].s4();
                ptr[i*16 + 5] = acc_buf[i + off].s5();
                ptr[i*16 + 6] = acc_buf[i + off].s6();
                ptr[i*16 + 7] = acc_buf[i + off].s7();
                ptr[i*16 + 8] = acc_buf[i + off].s8();
                ptr[i*16 + 9] = acc_buf[i + off].s9();
                ptr[i*16 +10] = acc_buf[i + off].sA();
                ptr[i*16 +11] = acc_buf[i + off].sB();
                ptr[i*16 +12] = acc_buf[i + off].sC();
                ptr[i*16 +13] = acc_buf[i + off].sD();
                ptr[i*16 +14] = acc_buf[i + off].sE();
                ptr[i*16 +15] = acc_buf[i + off].sF();
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    
    }

    template<class T>
    void flatten_build_from_usm(sycl::buffer<sycl::vec<T,16>> & buf_ret,T* usm_ptr, u64 len, u64 offset){

        auto ev = instance::get_compute_queue().submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {buf_ret, cgh, sycl::write_only, sycl::no_init};

            u64 off = offset;
            T* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){

                acc_buf[i + off].s0() = ptr[i*16 + 0];
                acc_buf[i + off].s1() = ptr[i*16 + 1];
                acc_buf[i + off].s2() = ptr[i*16 + 2];
                acc_buf[i + off].s3() = ptr[i*16 + 3];
                acc_buf[i + off].s4() = ptr[i*16 + 4];
                acc_buf[i + off].s5() = ptr[i*16 + 5];
                acc_buf[i + off].s6() = ptr[i*16 + 6];
                acc_buf[i + off].s7() = ptr[i*16 + 7];
                acc_buf[i + off].s8() = ptr[i*16 + 8];
                acc_buf[i + off].s9() = ptr[i*16 + 9];
                acc_buf[i + off].sA() = ptr[i*16 +10];
                acc_buf[i + off].sB() = ptr[i*16 +11];
                acc_buf[i + off].sC() = ptr[i*16 +12];
                acc_buf[i + off].sD() = ptr[i*16 +13];
                acc_buf[i + off].sE() = ptr[i*16 +14];
                acc_buf[i + off].sF() = ptr[i*16 +15];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls

    }
















    template<class T>
    void CommBuffer<sycl::buffer<T>,DirectGPUFlatten>::copy_to_usm(sycl::buffer<T> & obj_ref, u64 len, u64 offset){
        
        flatten_copy_to_usm(obj_ref,usm_ptr,len,offset);
    
    }

    template<class T>
    sycl::buffer<T> CommBuffer<sycl::buffer<T>,DirectGPUFlatten>::build_from_usm(u64 len, u64 offset){

        sycl::buffer<T> buf_ret (len);

        flatten_build_from_usm(buf_ret,usm_ptr,len,offset);

        return buf_ret;
    }

    template<class T>
    void CommBuffer<sycl::buffer<T>,DirectGPUFlatten>::isend(CommRequests & rqs, u32 rank_dest, u32 comm_tag, MPI_Comm comm){
        MPI_Request rq;
        mpi::isend(
            usm_ptr, 
            details.comm_len*int_len, 
            get_mpi_type<ptr_t>(), 
            rank_dest, 
            comm_tag, 
            comm, 
            &rq);
        rqs.push(rq);
    }

    template<class T>
    void CommBuffer<sycl::buffer<T>,DirectGPUFlatten>::irecv(CommRequests & rqs, u32 rank_src, u32 comm_tag, MPI_Comm comm){
        MPI_Request rq;
        mpi::irecv(usm_ptr, details.comm_len*int_len, get_mpi_type<ptr_t>(), rank_src, comm_tag, comm, &rq);
        rqs.push(rq);
    }


template<class T>
    CommBuffer<sycl::buffer<T>,DirectGPUFlatten> CommBuffer<sycl::buffer<T>,DirectGPUFlatten>::irecv_probe(CommRequests & rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm, CommDetails<sycl::buffer<T>> details){
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_src, comm_flag,comm, & st);
        mpi::get_count(&st, get_mpi_type<ptr_t>(), &cnt);

        if(cnt % int_len != 0){
            throw shamutils::throw_with_loc<std::runtime_error>("for this protocol the lenght of the received message must be a multiple of the number of components");
        }

        u32 val_cnt = cnt/int_len;

        CommDetails<sycl::buffer<T>> det {val_cnt};

        CommBuffer<sycl::buffer<T>,DirectGPUFlatten> ret {det};

        ret.irecv(rqs, rank_src, comm_flag, comm);

        return ret;
    }





































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