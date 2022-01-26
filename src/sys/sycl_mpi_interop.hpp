#include "mpi_handler.hpp"
#include <mpi.h>





inline MPI_Datatype mpi_type_i64 = MPI_LONG;
inline MPI_Datatype mpi_type_i32 = MPI_INT;
inline MPI_Datatype mpi_type_i16 = MPI_SHORT;
inline MPI_Datatype mpi_type_i8  = MPI_BYTE;
inline MPI_Datatype mpi_type_u64 = MPI_LONG;
inline MPI_Datatype mpi_type_u32 = MPI_INT;
inline MPI_Datatype mpi_type_u16 = MPI_SHORT;
inline MPI_Datatype mpi_type_u8  = MPI_BYTE;
inline MPI_Datatype mpi_type_f16 = MPI_SHORT; // no f16 in mpi std
inline MPI_Datatype mpi_type_f32 = MPI_FLOAT;
inline MPI_Datatype mpi_type_f64 = MPI_DOUBLE;

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


inline MPI_Datatype mpi_type_i64_2;
inline MPI_Datatype mpi_type_i32_2;
inline MPI_Datatype mpi_type_i16_2;
inline MPI_Datatype mpi_type_i8_2 ;
inline MPI_Datatype mpi_type_u64_2;
inline MPI_Datatype mpi_type_u32_2;
inline MPI_Datatype mpi_type_u16_2;
inline MPI_Datatype mpi_type_u8_2 ;
inline MPI_Datatype mpi_type_f16_2;
inline MPI_Datatype mpi_type_f32_2;
inline MPI_Datatype mpi_type_f64_2;

inline MPI_Datatype mpi_type_i64_3;
inline MPI_Datatype mpi_type_i32_3;
inline MPI_Datatype mpi_type_i16_3;
inline MPI_Datatype mpi_type_i8_3;
inline MPI_Datatype mpi_type_u64_3;
inline MPI_Datatype mpi_type_u32_3;
inline MPI_Datatype mpi_type_u16_3;
inline MPI_Datatype mpi_type_u8_3;
inline MPI_Datatype mpi_type_f16_3;
inline MPI_Datatype mpi_type_f32_3;
inline MPI_Datatype mpi_type_f64_3;

inline MPI_Datatype mpi_type_i64_4;
inline MPI_Datatype mpi_type_i32_4;
inline MPI_Datatype mpi_type_i16_4;
inline MPI_Datatype mpi_type_i8_4;
inline MPI_Datatype mpi_type_u64_4;
inline MPI_Datatype mpi_type_u32_4;
inline MPI_Datatype mpi_type_u16_4;
inline MPI_Datatype mpi_type_u8_4;
inline MPI_Datatype mpi_type_f16_4;
inline MPI_Datatype mpi_type_f32_4;
inline MPI_Datatype mpi_type_f64_4;

inline MPI_Datatype mpi_type_i64_8;
inline MPI_Datatype mpi_type_i32_8;
inline MPI_Datatype mpi_type_i16_8;
inline MPI_Datatype mpi_type_i8_8 ;
inline MPI_Datatype mpi_type_u64_8;
inline MPI_Datatype mpi_type_u32_8;
inline MPI_Datatype mpi_type_u16_8;
inline MPI_Datatype mpi_type_u8_8 ;
inline MPI_Datatype mpi_type_f16_8;
inline MPI_Datatype mpi_type_f32_8;
inline MPI_Datatype mpi_type_f64_8;

inline MPI_Datatype mpi_type_i64_16;
inline MPI_Datatype mpi_type_i32_16;
inline MPI_Datatype mpi_type_i16_16;
inline MPI_Datatype mpi_type_i8_16;
inline MPI_Datatype mpi_type_u64_16;
inline MPI_Datatype mpi_type_u32_16;
inline MPI_Datatype mpi_type_u16_16;
inline MPI_Datatype mpi_type_u8_16;
inline MPI_Datatype mpi_type_f16_16;
inline MPI_Datatype mpi_type_f32_16;
inline MPI_Datatype mpi_type_f64_16;




#define __SYCL_TYPE_COMMIT_len2(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.x()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec2, &offset_##base_name, & mpi_type_##src_type, &mpi_type_##base_name );\
    mpi::type_commit( &mpi_type_##base_name );\
}

#define __SYCL_TYPE_COMMIT_len3(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.x()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec3, &offset_##base_name, & mpi_type_##src_type, &__tmp_mpi_type_##base_name );\
    MPI_Aint lbs;\
    MPI_Aint exts;\
    mpi::type_get_extent(__tmp_mpi_type_##base_name, &lbs, &exts);\
    int sz_prim;\
    MPI_Type_size(mpi_type_##src_type, &sz_prim);\
    exts += sz_prim;\
    MPI_Type_create_resized(__tmp_mpi_type_##base_name, lbs, exts, &mpi_type_##base_name);\
    mpi::type_commit( &mpi_type_##base_name );\
}

#define __SYCL_TYPE_COMMIT_len4(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.x()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec4, &offset_##base_name, & mpi_type_##src_type, &mpi_type_##base_name );\
    mpi::type_commit( &mpi_type_##base_name );\
}

#define __SYCL_TYPE_COMMIT_len8(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.s0()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec8, &offset_##base_name, & mpi_type_##src_type, &mpi_type_##base_name );\
    mpi::type_commit( &mpi_type_##base_name );\
}

#define __SYCL_TYPE_COMMIT_len16(base_name,src_type)\
{\
    base_name a;\
    MPI_Aint offset_##base_name = ((size_t) ( (char *)&(a.s0()) - (char *)&(a) ));\
    mpi::type_create_struct( 1, & __len_vec16, &offset_##base_name, & mpi_type_##src_type, &mpi_type_##base_name );\
    mpi::type_commit( &mpi_type_##base_name );\
}


//TODO check mpi errors

inline void create_sycl_mpi_types(){


    
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


}

inline void free_sycl_mpi_types(){

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
}