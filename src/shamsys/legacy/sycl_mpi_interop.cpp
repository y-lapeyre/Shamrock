// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sycl_mpi_interop.hpp"
#include "shamsys/legacy/sycl_handler.hpp"





/*
const int __len_vec2  [] = {1,1};
const int __len_vec3  [] = {1,1,1};
const int __len_vec4  [] = {1,1,1,1};
const int __len_vec8  [] = {1,1,1,1,1,1,1,1};
const int __len_vec16 [] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
*/
const int __len_vec2 = 2;
const int __len_vec3 = 3;
const int __len_vec4 = 4;
const int __len_vec8 = 8;
const int __len_vec16 = 16;


inline MPI_Datatype __tmp_mpi_type_i64_3;
inline MPI_Datatype __tmp_mpi_type_i32_3;
inline MPI_Datatype __tmp_mpi_type_i16_3;
inline MPI_Datatype __tmp_mpi_type_i8_3; 
inline MPI_Datatype __tmp_mpi_type_u64_3;
inline MPI_Datatype __tmp_mpi_type_u32_3;
inline MPI_Datatype __tmp_mpi_type_u16_3;
inline MPI_Datatype __tmp_mpi_type_u8_3; 
inline MPI_Datatype __tmp_mpi_type_f16_3;
inline MPI_Datatype __tmp_mpi_type_f32_3;
inline MPI_Datatype __tmp_mpi_type_f64_3;




#define __SYCL_TYPE_COMMIT_len2(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.x()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec2, &offset_##base_name, & mpi_type_##src_type, &mpi_type_##base_name );\
    mpi::type_commit( &mpi_type_##base_name ); logger::debug_mpi_ln("SYCL MPI Interop", "init mpi type for : " #base_name );\
}

#define __SYCL_TYPE_COMMIT_len3(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.x()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec3, &offset_##base_name, & mpi_type_##src_type, &__tmp_mpi_type_##base_name );\
    mpi::type_create_resized(__tmp_mpi_type_##base_name, 0, sizeof(base_name), &mpi_type_##base_name);\
    mpi::type_commit( &mpi_type_##base_name ); logger::debug_mpi_ln("SYCL MPI Interop", "init mpi type for : " #base_name );\
}

#define __SYCL_TYPE_COMMIT_len4(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.x()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec4, &offset_##base_name, & mpi_type_##src_type, &mpi_type_##base_name );\
    mpi::type_commit( &mpi_type_##base_name ); logger::debug_mpi_ln("SYCL MPI Interop", "init mpi type for : " #base_name );\
}

#define __SYCL_TYPE_COMMIT_len8(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.s0()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec8, &offset_##base_name, & mpi_type_##src_type, &mpi_type_##base_name );\
    mpi::type_commit( &mpi_type_##base_name ); logger::debug_mpi_ln("SYCL MPI Interop", "init mpi type for : " #base_name );\
}

#define __SYCL_TYPE_COMMIT_len16(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.s0()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec16, &offset_##base_name, & mpi_type_##src_type, &mpi_type_##base_name );\
    mpi::type_commit( &mpi_type_##base_name ); logger::debug_mpi_ln("SYCL MPI Interop", "init mpi type for : " #base_name );\
}


//TODO check mpi errors

void create_sycl_mpi_types(){


    
    __SYCL_TYPE_COMMIT_len2(i64_2,i64)
    __SYCL_TYPE_COMMIT_len2(i32_2,i32)
    __SYCL_TYPE_COMMIT_len2(i16_2,i16)
    __SYCL_TYPE_COMMIT_len2(i8_2 ,i8 )
    __SYCL_TYPE_COMMIT_len2(u64_2,u64)
    __SYCL_TYPE_COMMIT_len2(u32_2,u32)
    __SYCL_TYPE_COMMIT_len2(u16_2,u16)
    __SYCL_TYPE_COMMIT_len2(u8_2 ,u8 )
    __SYCL_TYPE_COMMIT_len2(f16_2,f16)
    __SYCL_TYPE_COMMIT_len2(f32_2,f32)
    __SYCL_TYPE_COMMIT_len2(f64_2,f64)


    __SYCL_TYPE_COMMIT_len3(i64_3,i64)
    __SYCL_TYPE_COMMIT_len3(i32_3,i32)
    __SYCL_TYPE_COMMIT_len3(i16_3,i16)

    // {
    //     i16_3 a;

    //     MPI_Datatype types_list[3] = {mpi_type_i16,mpi_type_i16,mpi_type_i16};
    //     int          block_lens[3] = {1,1,1};
    //     MPI_Aint     MPI_offset[3];
    //        MPI_offset[0] = ((size_t) ( (char *)&(a.x()) - (char *)&(a) ));
    //        MPI_offset[1] = ((size_t) ( (char *)&(a.y()) - (char *)&(a) ));
    //        MPI_offset[2] = ((size_t) ( (char *)&(a.z()) - (char *)&(a) ));
        

    //     mpi::type_create_struct( 3,  block_lens, MPI_offset, types_list, &mpi_type_i16_3 );
    //     /*mpi::type_create_resized(__tmp_mpi_type_i16_3, 0, sizeof(base_name), &mpi_type_i16_3);*/\
    //     mpi::type_commit( &mpi_type_i16_3 );
    // }


    __SYCL_TYPE_COMMIT_len3(i8_3 ,i8 )
    __SYCL_TYPE_COMMIT_len3(u64_3,u64)
    __SYCL_TYPE_COMMIT_len3(u32_3,u32)
    __SYCL_TYPE_COMMIT_len3(u16_3,u16)
    __SYCL_TYPE_COMMIT_len3(u8_3 ,u8 )
    __SYCL_TYPE_COMMIT_len3(f16_3,f16)
    __SYCL_TYPE_COMMIT_len3(f32_3,f32)
    __SYCL_TYPE_COMMIT_len3(f64_3,f64)

    __SYCL_TYPE_COMMIT_len4(i64_4,i64)
    __SYCL_TYPE_COMMIT_len4(i32_4,i32)
    __SYCL_TYPE_COMMIT_len4(i16_4,i16)
    __SYCL_TYPE_COMMIT_len4(i8_4 ,i8 )
    __SYCL_TYPE_COMMIT_len4(u64_4,u64)
    __SYCL_TYPE_COMMIT_len4(u32_4,u32)
    __SYCL_TYPE_COMMIT_len4(u16_4,u16)
    __SYCL_TYPE_COMMIT_len4(u8_4 ,u8 )
    __SYCL_TYPE_COMMIT_len4(f16_4,f16)
    __SYCL_TYPE_COMMIT_len4(f32_4,f32)
    __SYCL_TYPE_COMMIT_len4(f64_4,f64)

    __SYCL_TYPE_COMMIT_len8(i64_8,i64)
    __SYCL_TYPE_COMMIT_len8(i32_8,i32)
    __SYCL_TYPE_COMMIT_len8(i16_8,i16)
    __SYCL_TYPE_COMMIT_len8(i8_8 ,i8 )
    __SYCL_TYPE_COMMIT_len8(u64_8,u64)
    __SYCL_TYPE_COMMIT_len8(u32_8,u32)
    __SYCL_TYPE_COMMIT_len8(u16_8,u16)
    __SYCL_TYPE_COMMIT_len8(u8_8 ,u8 )
    __SYCL_TYPE_COMMIT_len8(f16_8,f16)
    __SYCL_TYPE_COMMIT_len8(f32_8,f32)
    __SYCL_TYPE_COMMIT_len8(f64_8,f64)

    __SYCL_TYPE_COMMIT_len16(i64_16,i64)
    __SYCL_TYPE_COMMIT_len16(i32_16,i32)
    __SYCL_TYPE_COMMIT_len16(i16_16,i16)
    __SYCL_TYPE_COMMIT_len16(i8_16 ,i8 )
    __SYCL_TYPE_COMMIT_len16(u64_16,u64)
    __SYCL_TYPE_COMMIT_len16(u32_16,u32)
    __SYCL_TYPE_COMMIT_len16(u16_16,u16)
    __SYCL_TYPE_COMMIT_len16(u8_16 ,u8 )
    __SYCL_TYPE_COMMIT_len16(f16_16,f16)
    __SYCL_TYPE_COMMIT_len16(f32_16,f32)
    __SYCL_TYPE_COMMIT_len16(f64_16,f64)

    __mpi_sycl_type_active = true;



}

void free_sycl_mpi_types(){

    mpi::type_free(&mpi_type_i64_2);
    mpi::type_free(&mpi_type_i32_2);
    mpi::type_free(&mpi_type_i16_2);
    mpi::type_free(&mpi_type_i8_2 );
    mpi::type_free(&mpi_type_u64_2);
    mpi::type_free(&mpi_type_u32_2);
    mpi::type_free(&mpi_type_u16_2);
    mpi::type_free(&mpi_type_u8_2 );
    mpi::type_free(&mpi_type_f16_2);
    mpi::type_free(&mpi_type_f32_2);
    mpi::type_free(&mpi_type_f64_2);

    mpi::type_free(&mpi_type_i64_3);
    mpi::type_free(&mpi_type_i32_3);
    mpi::type_free(&mpi_type_i16_3);
    mpi::type_free(&mpi_type_i8_3);
    mpi::type_free(&mpi_type_u64_3);
    mpi::type_free(&mpi_type_u32_3);
    mpi::type_free(&mpi_type_u16_3);
    mpi::type_free(&mpi_type_u8_3);
    mpi::type_free(&mpi_type_f16_3);
    mpi::type_free(&mpi_type_f32_3);
    mpi::type_free(&mpi_type_f64_3);

    mpi::type_free(&mpi_type_i64_4);
    mpi::type_free(&mpi_type_i32_4);
    mpi::type_free(&mpi_type_i16_4);
    mpi::type_free(&mpi_type_i8_4);
    mpi::type_free(&mpi_type_u64_4);
    mpi::type_free(&mpi_type_u32_4);
    mpi::type_free(&mpi_type_u16_4);
    mpi::type_free(&mpi_type_u8_4);
    mpi::type_free(&mpi_type_f16_4);
    mpi::type_free(&mpi_type_f32_4);
    mpi::type_free(&mpi_type_f64_4);

    mpi::type_free(&mpi_type_i64_8);
    mpi::type_free(&mpi_type_i32_8);
    mpi::type_free(&mpi_type_i16_8);
    mpi::type_free(&mpi_type_i8_8 );
    mpi::type_free(&mpi_type_u64_8);
    mpi::type_free(&mpi_type_u32_8);
    mpi::type_free(&mpi_type_u16_8);
    mpi::type_free(&mpi_type_u8_8 );
    mpi::type_free(&mpi_type_f16_8);
    mpi::type_free(&mpi_type_f32_8);
    mpi::type_free(&mpi_type_f64_8);

    mpi::type_free(&mpi_type_i64_16);
    mpi::type_free(&mpi_type_i32_16);
    mpi::type_free(&mpi_type_i16_16);
    mpi::type_free(&mpi_type_i8_16);
    mpi::type_free(&mpi_type_u64_16);
    mpi::type_free(&mpi_type_u32_16);
    mpi::type_free(&mpi_type_u16_16);
    mpi::type_free(&mpi_type_u8_16);
    mpi::type_free(&mpi_type_f16_16);
    mpi::type_free(&mpi_type_f32_16);
    mpi::type_free(&mpi_type_f64_16);

    __mpi_sycl_type_active = false;
}



















namespace impl::copy_to_host {

    using namespace mpi_sycl_interop;

    namespace send {
        template <class T> T * init(std::unique_ptr<sycl::buffer<T>> &buf, u32 comm_sz) {

            T *comm_ptr = sycl::malloc_host<T>(comm_sz, sycl_handler::get_compute_queue());
            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_host", comm_sz, "->", reinterpret_cast<void *>(comm_ptr));


            if (comm_sz > 0) {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM");

                auto ker_copy = sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor acc{*buf, cgh, sycl::read_only};

                    T *ptr = comm_ptr;

                    cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) { ptr[item.get_linear_id()] = acc[item]; });
                });

                ker_copy.wait();
            } else {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM (skipped size=0)");
            }

            return comm_ptr;
        }

        template <class T> void finalize(T *comm_ptr) {
            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", reinterpret_cast<void *>(comm_ptr));

            sycl::free(comm_ptr, sycl_handler::get_compute_queue());
        }
    } // namespace send

    namespace recv {
        template <class T> T *init(u32 comm_sz) {
            T *comm_ptr = sycl::malloc_host<T>(comm_sz, sycl_handler::get_compute_queue());

            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_host", comm_sz);

            return comm_ptr;
        };

        template <class T> void finalize(std::unique_ptr<sycl::buffer<T>> &buf, T *comm_ptr, u32 comm_sz) {
            

            if (comm_sz > 0) {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer");

                auto ker_copy = sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor acc{*buf, cgh, sycl::write_only};

                    T *ptr = comm_ptr;

                    cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) { acc[item] = ptr[item.get_linear_id()]; });
                });

                ker_copy.wait();
            } else {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer (skipped size=0)");
            }

            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", reinterpret_cast<void *>(comm_ptr));

            sycl::free(comm_ptr, sycl_handler::get_compute_queue());
        }
    } // namespace recv

} // namespace impl::copy_to_host








namespace impl::directgpu {

    using namespace mpi_sycl_interop;

    namespace send {
        template <class T> T *init(std::unique_ptr<sycl::buffer<T>> &buf, u32 comm_sz) {

            T *comm_ptr = sycl::malloc_device<T>(comm_sz, sycl_handler::get_compute_queue());
            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_device", comm_sz, "->", comm_ptr);

            

            if (comm_sz > 0) {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM");

                auto ker_copy = sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor acc{*buf, cgh, sycl::read_only};

                    T *ptr = comm_ptr;

                    cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) { ptr[item.get_linear_id()] = acc[item]; });
                });

                ker_copy.wait();
            } else {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM (skipped size=0)");
            }

            return comm_ptr;
        }

        template <class T> void finalize(T *comm_ptr) {
            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", comm_ptr);

            sycl::free(comm_ptr, sycl_handler::get_compute_queue());
        }



    } // namespace send

    namespace recv {
        template <class T> T *init(u32 comm_sz) {
            T *comm_ptr = sycl::malloc_device<T>(comm_sz, sycl_handler::get_compute_queue());

            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_device", comm_sz);

            return comm_ptr;
        };

        template <class T> void finalize(std::unique_ptr<sycl::buffer<T>> &buf, T *comm_ptr, u32 comm_sz) {
            
            if (comm_sz > 0) {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer");

                auto ker_copy = sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor acc{*buf, cgh, sycl::write_only};

                    T *ptr = comm_ptr;

                    cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) { acc[item] = ptr[item.get_linear_id()]; });
                });

                ker_copy.wait();
            } else {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer (skipped size=0)");
            }

            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", comm_ptr);

            sycl::free(comm_ptr, sycl_handler::get_compute_queue());
        }
    } // namespace recv




} // namespace impl::directgpu
















namespace mpi_sycl_interop {

    comm_type current_mode = CopyToHost;

    template <class T>
    BufferMpiRequest<T>::BufferMpiRequest(std::unique_ptr<sycl::buffer<T>> &sycl_buf, comm_type comm_mode,
                                                          op_type comm_op, u32 comm_sz)
        : comm_mode(comm_mode), comm_op(comm_op), comm_sz(comm_sz), sycl_buf(sycl_buf) {

        logger::debug_mpi_ln("PatchDataField MPI Comm", "starting mpi sycl comm ", comm_sz, comm_op, comm_mode);

        if (comm_mode == CopyToHost && comm_op == Send) {

            comm_ptr = impl::copy_to_host::send::init<T>(sycl_buf, comm_sz);

        } else if (comm_mode == CopyToHost && comm_op == Recv_Probe) {

            comm_ptr = impl::copy_to_host::recv::init<T>(comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            comm_ptr = impl::directgpu::send::init<T>(sycl_buf, comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Recv_Probe) {

            comm_ptr = impl::directgpu::recv::init<T>(comm_sz);

        } 
        else {
            logger::err_ln("PatchDataField MPI Comm", "communication mode & op combination not implemented :", comm_mode,
                           comm_op);
        }
    }

    template <class T> void BufferMpiRequest<T>::finalize() {

        logger::debug_mpi_ln("PatchDataField MPI Comm", "finalizing mpi sycl comm ", comm_sz, comm_op, comm_mode);

        sycl_buf = std::make_unique<sycl::buffer<T>>(comm_sz);

        if (comm_mode == CopyToHost && comm_op == Send) {

            impl::copy_to_host::send::finalize<T>(comm_ptr);

        } else if (comm_mode == CopyToHost && comm_op == Recv_Probe) {

            impl::copy_to_host::recv::finalize<T>(sycl_buf, comm_ptr, comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            impl::directgpu::send::finalize<T>(comm_ptr);

        } else if (comm_mode == DirectGPU && comm_op == Recv_Probe) {

            impl::directgpu::recv::finalize<T>(sycl_buf, comm_ptr, comm_sz);

        } else {
            logger::err_ln("PatchDataField MPI Comm", "communication mode & op combination not implemented :", comm_mode,
                           comm_op);
        }
    }

#define X(a) template struct BufferMpiRequest<a>;
    XMAC_SYCLMPI_TYPE_ENABLED
#undef X

} // namespace mpi_sycl_interop