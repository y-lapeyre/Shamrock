#include "../shamrocktest.hpp"

#include "../../sys/mpi_handler.hpp"
#include "../../sys/sycl_handler.hpp"
#include "../../sys/sycl_mpi_interop.hpp"

#include <mpi.h>
#include <random>

/*
    f32_4 a {0,1,2,3};
    std::cout << ((size_t) ( (char *)&(a.x()) - (char *)&(a) )) << std::endl;
    std::cout << ((size_t) ( (char *)&(a.y()) - (char *)&(a) )) << std::endl;
    std::cout << ((size_t) ( (char *)&(a.z()) - (char *)&(a) )) << std::endl;
    std::cout << ((size_t) ( (char *)&(a.w()) - (char *)&(a) )) << std::endl;
*/

int get_mpi_size(MPI_Datatype md){
    MPI_Aint lbs;
    MPI_Aint exts;
    mpi::type_get_extent(md, &lbs, &exts);
    return exts;
}

Test_start("sycl_handler::",test_sycl_mpi_types,2){

    std::mt19937 eng(0x1111);        
    std::uniform_real_distribution<f64> distval(-1e9,1e9);

    create_sycl_mpi_types();

    Test_assert("mpi type i64    correct", get_mpi_size( mpi_type_i64 )== sizeof(i64 ));
    Test_assert("mpi type i32    correct", get_mpi_size( mpi_type_i32 )== sizeof(i32 ));
    Test_assert("mpi type i16    correct", get_mpi_size( mpi_type_i16 )== sizeof(i16 ));
    Test_assert("mpi type i8     correct", get_mpi_size( mpi_type_i8  )== sizeof(i8  ));
    Test_assert("mpi type u64    correct", get_mpi_size( mpi_type_u64 )== sizeof(u64 ));
    Test_assert("mpi type u32    correct", get_mpi_size( mpi_type_u32 )== sizeof(u32 ));
    Test_assert("mpi type u16    correct", get_mpi_size( mpi_type_u16 )== sizeof(u16 ));
    Test_assert("mpi type u8     correct", get_mpi_size( mpi_type_u8  )== sizeof(u8  ));
    Test_assert("mpi type f16    correct", get_mpi_size( mpi_type_f16 )== sizeof(f16 ));
    Test_assert("mpi type f32    correct", get_mpi_size( mpi_type_f32 )== sizeof(f32 ));
    Test_assert("mpi type f64    correct", get_mpi_size( mpi_type_f64 )== sizeof(f64 ));

    Test_assert("mpi type i64_2  correct", get_mpi_size( mpi_type_i64_2 )== sizeof(i64_2 ));
    Test_assert("mpi type i32_2  correct", get_mpi_size( mpi_type_i32_2 )== sizeof(i32_2 ));
    Test_assert("mpi type i16_2  correct", get_mpi_size( mpi_type_i16_2 )== sizeof(i16_2 ));
    Test_assert("mpi type i8_2   correct", get_mpi_size( mpi_type_i8_2  )== sizeof(i8_2  ));
    Test_assert("mpi type u64_2  correct", get_mpi_size( mpi_type_u64_2 )== sizeof(u64_2 ));
    Test_assert("mpi type u32_2  correct", get_mpi_size( mpi_type_u32_2 )== sizeof(u32_2 ));
    Test_assert("mpi type u16_2  correct", get_mpi_size( mpi_type_u16_2 )== sizeof(u16_2 ));
    Test_assert("mpi type u8_2   correct", get_mpi_size( mpi_type_u8_2  )== sizeof(u8_2  ));
    Test_assert("mpi type f16_2  correct", get_mpi_size( mpi_type_f16_2 )== sizeof(f16_2 ));
    Test_assert("mpi type f32_2  correct", get_mpi_size( mpi_type_f32_2 )== sizeof(f32_2 ));
    Test_assert("mpi type f64_2  correct", get_mpi_size( mpi_type_f64_2 )== sizeof(f64_2 ));
    Test_assert("mpi type i64_3  correct", get_mpi_size( mpi_type_i64_3 )== sizeof(i64_3 ));
    Test_assert("mpi type i32_3  correct", get_mpi_size( mpi_type_i32_3 )== sizeof(i32_3 ));
    Test_assert("mpi type i16_3  correct", get_mpi_size( mpi_type_i16_3 )== sizeof(i16_3 ));
    Test_assert("mpi type i8_3   correct", get_mpi_size( mpi_type_i8_3  )== sizeof(i8_3  ));
    Test_assert("mpi type u64_3  correct", get_mpi_size( mpi_type_u64_3 )== sizeof(u64_3 ));
    Test_assert("mpi type u32_3  correct", get_mpi_size( mpi_type_u32_3 )== sizeof(u32_3 ));
    Test_assert("mpi type u16_3  correct", get_mpi_size( mpi_type_u16_3 )== sizeof(u16_3 ));
    Test_assert("mpi type u8_3   correct", get_mpi_size( mpi_type_u8_3  )== sizeof(u8_3  ));
    Test_assert("mpi type f16_3  correct", get_mpi_size( mpi_type_f16_3 )== sizeof(f16_3 ));
    Test_assert("mpi type f32_3  correct", get_mpi_size( mpi_type_f32_3 )== sizeof(f32_3 ));
    Test_assert("mpi type f64_3  correct", get_mpi_size( mpi_type_f64_3 )== sizeof(f64_3 ));
    Test_assert("mpi type i64_4  correct", get_mpi_size( mpi_type_i64_4 )== sizeof(i64_4 ));
    Test_assert("mpi type i32_4  correct", get_mpi_size( mpi_type_i32_4 )== sizeof(i32_4 ));
    Test_assert("mpi type i16_4  correct", get_mpi_size( mpi_type_i16_4 )== sizeof(i16_4 ));
    Test_assert("mpi type i8_4   correct", get_mpi_size( mpi_type_i8_4  )== sizeof(i8_4  ));
    Test_assert("mpi type u64_4  correct", get_mpi_size( mpi_type_u64_4 )== sizeof(u64_4 ));
    Test_assert("mpi type u32_4  correct", get_mpi_size( mpi_type_u32_4 )== sizeof(u32_4 ));
    Test_assert("mpi type u16_4  correct", get_mpi_size( mpi_type_u16_4 )== sizeof(u16_4 ));
    Test_assert("mpi type u8_4   correct", get_mpi_size( mpi_type_u8_4  )== sizeof(u8_4  ));
    Test_assert("mpi type f16_4  correct", get_mpi_size( mpi_type_f16_4 )== sizeof(f16_4 ));
    Test_assert("mpi type f32_4  correct", get_mpi_size( mpi_type_f32_4 )== sizeof(f32_4 ));
    Test_assert("mpi type f64_4  correct", get_mpi_size( mpi_type_f64_4 )== sizeof(f64_4 ));
    Test_assert("mpi type i64_8  correct", get_mpi_size( mpi_type_i64_8 )== sizeof(i64_8 ));
    Test_assert("mpi type i32_8  correct", get_mpi_size( mpi_type_i32_8 )== sizeof(i32_8 ));
    Test_assert("mpi type i16_8  correct", get_mpi_size( mpi_type_i16_8 )== sizeof(i16_8 ));
    Test_assert("mpi type i8_8   correct", get_mpi_size( mpi_type_i8_8  )== sizeof(i8_8  ));
    Test_assert("mpi type u64_8  correct", get_mpi_size( mpi_type_u64_8 )== sizeof(u64_8 ));
    Test_assert("mpi type u32_8  correct", get_mpi_size( mpi_type_u32_8 )== sizeof(u32_8 ));
    Test_assert("mpi type u16_8  correct", get_mpi_size( mpi_type_u16_8 )== sizeof(u16_8 ));
    Test_assert("mpi type u8_8   correct", get_mpi_size( mpi_type_u8_8  )== sizeof(u8_8  ));
    Test_assert("mpi type f16_8  correct", get_mpi_size( mpi_type_f16_8 )== sizeof(f16_8 ));
    Test_assert("mpi type f32_8  correct", get_mpi_size( mpi_type_f32_8 )== sizeof(f32_8 ));
    Test_assert("mpi type f64_8  correct", get_mpi_size( mpi_type_f64_8 )== sizeof(f64_8 ));
    Test_assert("mpi type i64_16 correct", get_mpi_size( mpi_type_i64_16)== sizeof(i64_16));
    Test_assert("mpi type i32_16 correct", get_mpi_size( mpi_type_i32_16)== sizeof(i32_16));
    Test_assert("mpi type i16_16 correct", get_mpi_size( mpi_type_i16_16)== sizeof(i16_16));
    Test_assert("mpi type i8_16  correct", get_mpi_size( mpi_type_i8_16 )== sizeof(i8_16 ));
    Test_assert("mpi type u64_16 correct", get_mpi_size( mpi_type_u64_16)== sizeof(u64_16));
    Test_assert("mpi type u32_16 correct", get_mpi_size( mpi_type_u32_16)== sizeof(u32_16));
    Test_assert("mpi type u16_16 correct", get_mpi_size( mpi_type_u16_16)== sizeof(u16_16));
    Test_assert("mpi type u8_16  correct", get_mpi_size( mpi_type_u8_16 )== sizeof(u8_16 ));
    Test_assert("mpi type f16_16 correct", get_mpi_size( mpi_type_f16_16)== sizeof(f16_16));
    Test_assert("mpi type f32_16 correct", get_mpi_size( mpi_type_f32_16)== sizeof(f32_16));
    Test_assert("mpi type f64_16 correct", get_mpi_size( mpi_type_f64_16)== sizeof(f64_16));

    free_sycl_mpi_types();

}

