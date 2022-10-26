// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sycl_mpi_interop.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once


#include "mpi_handler.hpp"
#include "log.hpp"


inline bool __mpi_sycl_type_active = false;
inline bool is_mpi_sycl_interop_active(){
    return __mpi_sycl_type_active;
}


inline MPI_Datatype mpi_type_i64 = MPI_INT64_T;
inline MPI_Datatype mpi_type_i32 = MPI_INT32_T;
inline MPI_Datatype mpi_type_i16 = MPI_INT16_T;
inline MPI_Datatype mpi_type_i8  = MPI_INT8_T ;
inline MPI_Datatype mpi_type_u64 = MPI_UINT64_T;
inline MPI_Datatype mpi_type_u32 = MPI_UINT32_T;
inline MPI_Datatype mpi_type_u16 = MPI_UINT16_T;
inline MPI_Datatype mpi_type_u8  = MPI_UINT8_T ;
inline MPI_Datatype mpi_type_f16 = MPI_SHORT; // no f16 in mpi std
inline MPI_Datatype mpi_type_f32 = MPI_FLOAT;
inline MPI_Datatype mpi_type_f64 = MPI_DOUBLE;


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
inline MPI_Datatype mpi_type_i8_3 ; 
inline MPI_Datatype mpi_type_u64_3;
inline MPI_Datatype mpi_type_u32_3;
inline MPI_Datatype mpi_type_u16_3;
inline MPI_Datatype mpi_type_u8_3 ; 
inline MPI_Datatype mpi_type_f16_3;
inline MPI_Datatype mpi_type_f32_3;
inline MPI_Datatype mpi_type_f64_3;

inline MPI_Datatype mpi_type_i64_4;
inline MPI_Datatype mpi_type_i32_4;
inline MPI_Datatype mpi_type_i16_4;
inline MPI_Datatype mpi_type_i8_4 ; 
inline MPI_Datatype mpi_type_u64_4;
inline MPI_Datatype mpi_type_u32_4;
inline MPI_Datatype mpi_type_u16_4;
inline MPI_Datatype mpi_type_u8_4 ; 
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
inline MPI_Datatype mpi_type_i8_16 ;
inline MPI_Datatype mpi_type_u64_16;
inline MPI_Datatype mpi_type_u32_16;
inline MPI_Datatype mpi_type_u16_16;
inline MPI_Datatype mpi_type_u8_16 ;
inline MPI_Datatype mpi_type_f16_16;
inline MPI_Datatype mpi_type_f32_16;
inline MPI_Datatype mpi_type_f64_16;


template<class type>
MPI_Datatype & get_mpi_type(); 

//coments due to weird implementation of half prec in hipsycl

template<> inline MPI_Datatype & get_mpi_type<i64      >(){return mpi_type_i64    ;}
template<> inline MPI_Datatype & get_mpi_type<i32      >(){return mpi_type_i32    ;}
template<> inline MPI_Datatype & get_mpi_type<i16      >(){return mpi_type_i16    ;}
template<> inline MPI_Datatype & get_mpi_type<i8       >(){return mpi_type_i8     ;}
template<> inline MPI_Datatype & get_mpi_type<u64      >(){return mpi_type_u64    ;}
template<> inline MPI_Datatype & get_mpi_type<u32      >(){return mpi_type_u32    ;}
template<> inline MPI_Datatype & get_mpi_type<u16      >(){return mpi_type_u16    ;}
template<> inline MPI_Datatype & get_mpi_type<u8       >(){return mpi_type_u8     ;}
//template<> inline MPI_Datatype & get_mpi_type<f16      >(){return mpi_type_f16    ;}
template<> inline MPI_Datatype & get_mpi_type<f32      >(){return mpi_type_f32    ;}
template<> inline MPI_Datatype & get_mpi_type<f64      >(){return mpi_type_f64    ;}

template<> inline MPI_Datatype & get_mpi_type<i64_2    >(){return mpi_type_i64_2  ;}
template<> inline MPI_Datatype & get_mpi_type<i32_2    >(){return mpi_type_i32_2  ;}
template<> inline MPI_Datatype & get_mpi_type<i16_2    >(){return mpi_type_i16_2  ;}
template<> inline MPI_Datatype & get_mpi_type<i8_2     >(){return mpi_type_i8_2   ;}
template<> inline MPI_Datatype & get_mpi_type<u64_2    >(){return mpi_type_u64_2  ;}
template<> inline MPI_Datatype & get_mpi_type<u32_2    >(){return mpi_type_u32_2  ;}
template<> inline MPI_Datatype & get_mpi_type<u16_2    >(){return mpi_type_u16_2  ;}
template<> inline MPI_Datatype & get_mpi_type<u8_2     >(){return mpi_type_u8_2   ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_2    >(){return mpi_type_f16_2  ;}
template<> inline MPI_Datatype & get_mpi_type<f32_2    >(){return mpi_type_f32_2  ;}
template<> inline MPI_Datatype & get_mpi_type<f64_2    >(){return mpi_type_f64_2  ;}

template<> inline MPI_Datatype & get_mpi_type<i64_3    >(){return mpi_type_i64_3  ;}
template<> inline MPI_Datatype & get_mpi_type<i32_3    >(){return mpi_type_i32_3  ;}
template<> inline MPI_Datatype & get_mpi_type<i16_3    >(){return mpi_type_i16_3  ;}
template<> inline MPI_Datatype & get_mpi_type<i8_3     >(){return mpi_type_i8_3   ;}
template<> inline MPI_Datatype & get_mpi_type<u64_3    >(){return mpi_type_u64_3  ;}
template<> inline MPI_Datatype & get_mpi_type<u32_3    >(){return mpi_type_u32_3  ;}
template<> inline MPI_Datatype & get_mpi_type<u16_3    >(){return mpi_type_u16_3  ;}
template<> inline MPI_Datatype & get_mpi_type<u8_3     >(){return mpi_type_u8_3   ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_3    >(){return mpi_type_f16_3  ;}
template<> inline MPI_Datatype & get_mpi_type<f32_3    >(){return mpi_type_f32_3  ;}
template<> inline MPI_Datatype & get_mpi_type<f64_3    >(){return mpi_type_f64_3  ;}

template<> inline MPI_Datatype & get_mpi_type<i64_4    >(){return mpi_type_i64_4  ;}
template<> inline MPI_Datatype & get_mpi_type<i32_4    >(){return mpi_type_i32_4  ;}
template<> inline MPI_Datatype & get_mpi_type<i16_4    >(){return mpi_type_i16_4  ;}
template<> inline MPI_Datatype & get_mpi_type<i8_4     >(){return mpi_type_i8_4   ;}
template<> inline MPI_Datatype & get_mpi_type<u64_4    >(){return mpi_type_u64_4  ;}
template<> inline MPI_Datatype & get_mpi_type<u32_4    >(){return mpi_type_u32_4  ;}
template<> inline MPI_Datatype & get_mpi_type<u16_4    >(){return mpi_type_u16_4  ;}
template<> inline MPI_Datatype & get_mpi_type<u8_4     >(){return mpi_type_u8_4   ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_4    >(){return mpi_type_f16_4  ;}
template<> inline MPI_Datatype & get_mpi_type<f32_4    >(){return mpi_type_f32_4  ;}
template<> inline MPI_Datatype & get_mpi_type<f64_4    >(){return mpi_type_f64_4  ;}

template<> inline MPI_Datatype & get_mpi_type<i64_8    >(){return mpi_type_i64_8  ;}
template<> inline MPI_Datatype & get_mpi_type<i32_8    >(){return mpi_type_i32_8  ;}
template<> inline MPI_Datatype & get_mpi_type<i16_8    >(){return mpi_type_i16_8  ;}
template<> inline MPI_Datatype & get_mpi_type<i8_8     >(){return mpi_type_i8_8   ;}
template<> inline MPI_Datatype & get_mpi_type<u64_8    >(){return mpi_type_u64_8  ;}
template<> inline MPI_Datatype & get_mpi_type<u32_8    >(){return mpi_type_u32_8  ;}
template<> inline MPI_Datatype & get_mpi_type<u16_8    >(){return mpi_type_u16_8  ;}
template<> inline MPI_Datatype & get_mpi_type<u8_8     >(){return mpi_type_u8_8   ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_8    >(){return mpi_type_f16_8  ;}
template<> inline MPI_Datatype & get_mpi_type<f32_8    >(){return mpi_type_f32_8  ;}
template<> inline MPI_Datatype & get_mpi_type<f64_8    >(){return mpi_type_f64_8  ;}

template<> inline MPI_Datatype & get_mpi_type<i64_16   >(){return mpi_type_i64_16 ;}
template<> inline MPI_Datatype & get_mpi_type<i32_16   >(){return mpi_type_i32_16 ;}
template<> inline MPI_Datatype & get_mpi_type<i16_16   >(){return mpi_type_i16_16 ;}
template<> inline MPI_Datatype & get_mpi_type<i8_16    >(){return mpi_type_i8_16  ;}
template<> inline MPI_Datatype & get_mpi_type<u64_16   >(){return mpi_type_u64_16 ;}
template<> inline MPI_Datatype & get_mpi_type<u32_16   >(){return mpi_type_u32_16 ;}
template<> inline MPI_Datatype & get_mpi_type<u16_16   >(){return mpi_type_u16_16 ;}
template<> inline MPI_Datatype & get_mpi_type<u8_16    >(){return mpi_type_u8_16  ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_16   >(){return mpi_type_f16_16 ;}
template<> inline MPI_Datatype & get_mpi_type<f32_16   >(){return mpi_type_f32_16 ;}
template<> inline MPI_Datatype & get_mpi_type<f64_16   >(){return mpi_type_f64_16 ;}









void create_sycl_mpi_types();

void free_sycl_mpi_types();






#define XMAC_COMM_TYPE_ENABLED \
    X(f32   ) \
    X(f32_2 ) \
    X(f32_3 ) \
    X(f32_4 ) \
    X(f32_8 ) \
    X(f32_16) \
    X(f64   ) \
    X(f64_2 ) \
    X(f64_3 ) \
    X(f64_4 ) \
    X(f64_8 ) \
    X(f64_16) \
    X(u8   )  \
    X(u32   ) \
    X(u64   )





namespace mpi_sycl_interop {



    enum comm_type {
        CopyToHost, DirectGPU
    };
    enum op_type{
        Send,Recv
    };

    extern comm_type current_mode;



    template<class T>
    struct BufferMpiRequest{

        static constexpr bool is_in_type_list = 
            #define X(args)  std::is_same<T, args>::value ||
            XMAC_COMM_TYPE_ENABLED false
            #undef X
            ;

        static_assert(is_in_type_list
            , "BufferMpiRequest must be one of those types : "

            #define X(args) #args " "
            XMAC_COMM_TYPE_ENABLED
            #undef X
            );


        MPI_Request mpi_rq;
        comm_type comm_mode;
        op_type comm_op;
        T* comm_ptr;
        u32 comm_sz;
        std::unique_ptr<sycl::buffer<T>> &pdat_field;


        BufferMpiRequest<T> (
            std::unique_ptr<sycl::buffer<T>> &pdat_field,
            comm_type comm_mode,
            op_type comm_op,
            u32 comm_sz
            );

        inline T* get_mpi_ptr(){
            return comm_ptr;
        }

        void finalize();
    };


    template<class T>
    inline u64 isend( std::unique_ptr<sycl::buffer<T>> &p, const u32 &size_comm, std::vector<BufferMpiRequest<T>> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm){

        rq_lst.push_back(BufferMpiRequest<T>(p,current_mode,Send,size_comm));
            
        u32 rq_index = rq_lst.size() - 1;

        auto & rq = rq_lst[rq_index];   

        mpi::isend(rq.get_mpi_ptr(), size_comm, get_mpi_type<T>(), rank_dest, tag, comm, &(rq_lst[rq_index].mpi_rq));
        
        return sizeof(T)*size_comm;
    }


    template<class T>
    inline u64 irecv(std::unique_ptr<sycl::buffer<T>> &p, const u32 &size_comm, std::vector<BufferMpiRequest<T>> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){
        
        rq_lst.push_back(BufferMpiRequest<T>(p,current_mode,Recv,size_comm));
            
        u32 rq_index = rq_lst.size() - 1;

        auto & rq = rq_lst[rq_index];   

        mpi::irecv(rq.get_mpi_ptr(), size_comm, get_mpi_type<T>(), rank_source, tag, comm, &(rq_lst[rq_index].mpi_rq));

        return sizeof(T)*size_comm;
    }

    template<class T>
    inline u64 irecv_probe(std::unique_ptr<sycl::buffer<T>> &p, std::vector<BufferMpiRequest<T>> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, get_mpi_type<T>(), &cnt);

        u32 len = cnt ;

        return irecv(p, len, rq_lst, rank_source,tag,comm);
    }


    

    template<class T> 
    inline std::vector<MPI_Request> get_rqs(std::vector<BufferMpiRequest<T>> &rq_lst){
        std::vector<MPI_Request> addrs;

        for(auto a : rq_lst){
            addrs.push_back(a.mpi_rq);
        }

        return addrs;
    }

    template<class T>
    inline void waitall(std::vector<BufferMpiRequest<T>> &rq_lst){
        std::vector<MPI_Request> addrs;

        for(auto a : rq_lst){
            addrs.push_back(a.mpi_rq);
        }

        std::vector<MPI_Status> st_lst(addrs.size());
        mpi::waitall(addrs.size(), addrs.data(), st_lst.data());

        for(auto a : rq_lst){
            a.finalize();
        }
    }



    template<class T>
    inline void file_write(MPI_File fh, std::unique_ptr<sycl::buffer<T>> &p, const u32 &size_comm){
        MPI_Status st;

        BufferMpiRequest<T> rq (p, current_mode, Send,size_comm);

        mpi::file_write(fh, rq.get_mpi_ptr(),  size_comm, get_mpi_type<T>(), &st);

        rq.finalize();
    }



}