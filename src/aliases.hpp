/**
 * @file aliases.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief typedefs and macros
 * @date 2021-09-17
 * @copyright Copyright Timothée David--Cléris (c) 2021
 * 
 */


#pragma once

#include <string.h>
#include <sycl/sycl.hpp>


#define __FILENAME__ std::string(strstr(__FILE__, "/src/") ? strstr(__FILE__, "/src/")+1  : __FILE__)
#define __LOC_PREFIX__  __FILENAME__ +":" + std::to_string(__LINE__)
//#define throw_with_pos(...) throw std::runtime_error( __VA_ARGS__ " ("+ __FILENAME__ +":" + std::to_string(__LINE__) +")");

//#define PTR_FREE(...)      {if(__VA_ARGS__ != NULL){ delete   __VA_ARGS__; __VA_ARGS__ = NULL; }else{ throw_with_pos("trying to free \"" #__VA_ARGS__ "\" but it was already free'd");}}
//#define PTR_FREE_ARR(...)  {if(__VA_ARGS__ != NULL){ delete[] __VA_ARGS__; __VA_ARGS__ = NULL; }else{ throw_with_pos("trying to free array \"" #__VA_ARGS__ "\" but it was already free'd");}}

#ifdef SYCL_COMP_HIPSYCL
typedef sycl::detail::s_long   i64;
typedef sycl::detail::s_int    i32;
typedef sycl::detail::s_short  i16;
typedef sycl::detail::s_char   i8 ;
typedef sycl::detail::u_long   u64;
typedef sycl::detail::u_int    u32;
typedef sycl::detail::u_short  u16;
typedef sycl::detail::u_char   u8 ;
typedef sycl::detail::hp_float f16;
typedef sycl::detail::sp_float f32;
typedef sycl::detail::dp_float f64;
#endif

#ifdef SYCL_COMP_DPCPP
using i64 = std::int64_t  ;
using i32 = std::int32_t  ;
using i16 = std::int16_t  ;
using i8  = std::int8_t   ;
using u64 = std::uint64_t ;
using u32 = std::uint32_t ;
using u16 = std::uint16_t ;
using u8  = std::uint8_t  ;
using f16 = sycl::half    ;
using f32 = float         ;
using f64 = double        ;
#endif


constexpr i64 i64_max = 0x7FFFFFFFFFFFFFFF;
constexpr i32 i32_max = 0x7FFFFFFF;
constexpr i16 i16_max = 0x7FFF;
constexpr i8  i8_max  = 0x7F;

constexpr i64 i64_min = 0x8000000000000000;
constexpr i32 i32_min = 0x80000000;
constexpr i16 i16_min = 0x8000;
constexpr i8  i8_min  = 0x80;

constexpr u64 u64_max = 0xFFFFFFFFFFFFFFFF;
constexpr u32 u32_max = 0xFFFFFFFF;
constexpr u16 u16_max = 0xFFFF;
constexpr u8  u8_max  = 0xFF;

constexpr u64 u64_min = 0x0000000000000000;
constexpr u32 u32_min = 0x00000000;
constexpr u16 u16_min = 0x0000;
constexpr u8  u8_min  = 0x00;


#define TYPEDEFS_TYPES(...) \
using i64_##__VA_ARGS__ = sycl::vec<i64,__VA_ARGS__>;\
using i32_##__VA_ARGS__ = sycl::vec<i32,__VA_ARGS__>;\
using i16_##__VA_ARGS__ = sycl::vec<i16,__VA_ARGS__>;\
using i8_##__VA_ARGS__  = sycl::vec<i8 ,__VA_ARGS__>;\
using u64_##__VA_ARGS__ = sycl::vec<u64,__VA_ARGS__>;\
using u32_##__VA_ARGS__ = sycl::vec<u32,__VA_ARGS__>;\
using u16_##__VA_ARGS__ = sycl::vec<u16,__VA_ARGS__>;\
using u8_##__VA_ARGS__  = sycl::vec<u8 ,__VA_ARGS__>;\
using f16_##__VA_ARGS__ = sycl::vec<f16,__VA_ARGS__>;\
using f32_##__VA_ARGS__ = sycl::vec<f32,__VA_ARGS__>;\
using f64_##__VA_ARGS__ = sycl::vec<f64,__VA_ARGS__>;\

TYPEDEFS_TYPES(2)
TYPEDEFS_TYPES(3)
TYPEDEFS_TYPES(4)
TYPEDEFS_TYPES(8)
TYPEDEFS_TYPES(16)




#define ERR_ID_64 18446744073709551615u


extern std::string git_info_str;




const u32 term_width = 64;

inline std::string shamrock_title_bar_big = "\n\
  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████\n\
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░ \n\
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███   \n\
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████    \n\
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███   \n\
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███  \n\
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████\n\
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░ \n\
";

inline void print_title_bar(){
    printf("%s\n",shamrock_title_bar_big.c_str());
    printf("---------------------------------------------------------------------------------");
    printf("%s\n",git_info_str.c_str());
    printf("---------------------------------------------------------------------------------\n");

}