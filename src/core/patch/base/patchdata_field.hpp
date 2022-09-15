// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>



#include "access/access.hpp"
#include "aliases.hpp"

#include "core/patch/base/enabled_fields.hpp"
#include "core/sys/log.hpp"
#include "core/sys/sycl_handler.hpp"
#include "core/sys/sycl_mpi_interop.hpp"
#include "core/utils/string_utils.hpp"
#include "core/utils/sycl_vector_utils.hpp"
#include "core/sys/mpi_handler.hpp"

#include "core/utils/sycl_algs.hpp"



template<class T>
class PatchDataField {

    static constexpr bool isprimitive = 
        std::is_same<T, f32>::value ||
        std::is_same<T, f64>::value ||
        std::is_same<T, u32>::value ||
        std::is_same<T, u64>::value;

    static constexpr bool is_in_type_list = 
        #define X(args)  std::is_same<T, args>::value ||
        XMAC_LIST_ENABLED_FIELD true
        #undef X
        ;

    static_assert(is_in_type_list
        , "PatchDataField must be one of those types : "

        #define X(args) #args " "
        XMAC_LIST_ENABLED_FIELD
        #undef X
        );

    using EnableIfPrimitive = enable_if_t<isprimitive>;

    using EnableIfVec = enable_if_t< is_in_type_list && (!isprimitive)>;





    std::unique_ptr<sycl::buffer<T>> buf;

    std::string field_name;

    u32 nvar; //number of variable per object
    u32 obj_cnt; // number of contained object

    u32 val_cnt; // nvar*obj_cnt

    u32 capacity;

    constexpr static u32 min_capa = 100;
    constexpr static f32 safe_fact = 1.25;


    void _alloc(){
        buf = std::make_unique<sycl::buffer<T>>(capacity);
        
        logger::debug_alloc_ln("PatchDataField", "allocate field :" , "len =",capacity);
    }

    void _free(){

        if(buf){
            logger::debug_alloc_ln("PatchDataField", "free field :" , "len =",capacity);

            buf.reset();
        }
    }

    public:

    //TODO find a way to add particles easily cf setup require public vector
    

    using Field_type = T;




    inline PatchDataField(std::string name, u32 nvar) : field_name(name) , nvar(nvar), obj_cnt(0), val_cnt(0), capacity(0){
        
        //_alloc();

    };


    PatchDataField(const PatchDataField& other) : field_name  (other.field_name)  ,
        nvar        (other.nvar      )  , 
        obj_cnt     (other.obj_cnt   )  ,
        val_cnt     (other.val_cnt   )  , 
        capacity    (other.capacity  )  {

        ;

        //field_data = other.field_data;

        if (capacity != 0) {
            _alloc();
            //copydata(other._data,_data, capacity);
            syclalgs::basic::copybuf_discard(*other.buf, *buf, capacity);
        }
        
    }


    PatchDataField(PatchDataField &&other) = delete;
    /* : _data(other._data){ 
        other._data = nullptr; 
    }*/

    PatchDataField &operator=(const PatchDataField &other) = delete;

    PatchDataField &operator=(PatchDataField &&other) =delete;
    
    /*noexcept {
        if (&other != this) {
            _free();
            _data       = other._data;
            other._data = nullptr;
        }
        return *this;
    }*/
    

    inline ~PatchDataField(){
        logger::debug_alloc_ln("PatchDataField", "free field :" , "len =",capacity);
        _free();
    }




    inline std::unique_ptr<sycl::buffer<T>> & get_buf(){
        return buf;
    }

    //[[deprecated]]
    //inline std::unique_ptr<sycl::buffer<T>> get_sub_buf(){
    //    if(capacity > 0){
    //        return std::make_unique<sycl::buffer<T>>(*buf,0,val_cnt);
    //    }
    //    return std::unique_ptr<sycl::buffer<T>>();
    //}




    inline u32 size(){
        return val_cnt;
    }

    inline u32 get_nvar(){
        return nvar;
    }

    inline u32 get_obj_cnt(){
        return obj_cnt;
    }

    inline std::string get_name(){
        return field_name;
    }

    //TODO add overflow check
    inline void resize(u32 new_obj_cnt){

        logger::debug_alloc_ln("PatchDataField", "resize from : ",val_cnt, "to :",new_obj_cnt*nvar);

        u32 new_size = new_obj_cnt*nvar;
        //field_data.resize(new_size);

        //*
        if(capacity == 0){
            capacity = safe_fact*new_size;
            _alloc();
        }else if (new_size > capacity) {
            
            //u32 old_capa = capacity;
            capacity = safe_fact*new_size;

            sycl::buffer<T>* old_buf = buf.release();


            _alloc();

            


  
            syclalgs::basic::copybuf_discard(*old_buf, *buf, val_cnt);


            logger::debug_alloc_ln("PatchDataField", "delete old buf : ");
            delete old_buf;
        }else{
            
        }
        //*/



        obj_cnt = new_obj_cnt;
        val_cnt = new_obj_cnt*nvar;
    }

    inline void expand(u32 obj_to_add){
        resize(obj_cnt + obj_to_add);
    }

    inline void shrink(u32 obj_to_rem){
        resize(obj_cnt - obj_to_rem);
    }



    void insert_element(T v);

    void apply_offset(T off);

    void insert(PatchDataField<T> &f2);

    

    inline void overwrite(PatchDataField<T> &f2, u32 obj_cnt){
        if(val_cnt < obj_cnt){
            throw shamrock_exc("to overwrite you need more element in the field");
        }

        {
            sycl::host_accessor acc {*buf};
            sycl::host_accessor acc_f2{*f2.get_buf()};

            for (u32 i = 0; i < obj_cnt; i++) {
                //field_data[idx_st + i] = f2.field_data[i];
                acc[i] = acc_f2[i];
            }

        }
    }

    inline void override(sycl::buffer<T> & data, u32 cnt){

        if(cnt !=  val_cnt) throw shamrock_exc("buffer size doesn't match patchdata field size"); // TODO remove ref to size

        

        if(val_cnt > 0) {

            {
                sycl::host_accessor acc_cur {*buf};
                auto acc = data.template get_access<sycl::access::mode::read>();

                for(u32 i = 0; i < val_cnt ; i++){
                    //field_data[i] = acc[i];
                    acc_cur[i] = acc[i];
                }

            }
        }
        
    }

    inline void override(const T val){

        if(val_cnt > 0) {

            {
                sycl::host_accessor acc {*buf};
                for(u32 i = 0; i < val_cnt ; i++){
                    //field_data[i] = val;
                    acc[i] = val;
                }

            }
        }
        
    }

    // use only if nvar = 1
    template<class Lambdacd>
    inline std::vector<u32> get_elements_with_range(Lambdacd && cd_true, T vmin, T vmax){
        std::vector<u32> idxs;

        {
            sycl::host_accessor acc {*buf};

            for(u32 i = 0; i < val_cnt ; i++){
                if (cd_true(acc[i], vmin, vmax)) {
                    idxs.push_back(i);
                }
            }

        }
        
        return idxs;
    }

    void extract_element(u32 pidx, PatchDataField<T> & to);

    bool check_field_match(PatchDataField<T> &f2);

    /**
     * @brief Copy all objects in idxs to pfield
     * 
     * @param idxs 
     * @param pfield 
     */
    void append_subset_to(std::vector<u32> & idxs, PatchDataField & pfield);


    void gen_mock_data(u32 obj_cnt, std::mt19937& eng);



};

namespace patchdata_field {

    enum comm_type {
        CopyToHost, DirectGPU
    };
    enum op_type{
        Send,Recv
    };

    extern comm_type current_mode;


    template<class T>
    struct PatchDataFieldMpiRequest{
        MPI_Request mpi_rq;
        comm_type comm_mode;
        op_type comm_op;
        T* comm_ptr;
        u32 comm_sz;
        PatchDataField<T> &pdat_field;


        PatchDataFieldMpiRequest<T> (
            PatchDataField<T> &pdat_field,
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
    inline u64 isend( PatchDataField<T> &p, std::vector<PatchDataFieldMpiRequest<T>> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm){

        u32 size_comm = p.size();

        rq_lst.emplace_back(p,current_mode,Send,size_comm);
            
        u32 rq_index = rq_lst.size() - 1;

        auto & rq = rq_lst[rq_index];   

        mpi::isend(rq.get_mpi_ptr(), size_comm, get_mpi_type<T>(), rank_dest, tag, comm, &(rq_lst[rq_index].mpi_rq));
        
        return sizeof(T)*p.size();
    }

    template<class T>
    inline u64 irecv(PatchDataField<T> &p, std::vector<PatchDataFieldMpiRequest<T>> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, get_mpi_type<T>(), &cnt);

        u32 len = cnt / p.get_nvar();



        rq_lst.emplace_back(p,current_mode,Recv,len);
            
        u32 rq_index = rq_lst.size() - 1;

        auto & rq = rq_lst[rq_index];   

        mpi::irecv(rq.get_mpi_ptr(), cnt, get_mpi_type<T>(), rank_source, tag, comm, &(rq_lst[rq_index].mpi_rq));

        return sizeof(T)*cnt;
    }

    template<class T> 
    inline std::vector<MPI_Request> get_rqs(std::vector<PatchDataFieldMpiRequest<T>> &rq_lst){
        std::vector<MPI_Request> addrs;

        for(auto a : rq_lst){
            addrs.push_back(a.mpi_rq);
        }

        return addrs;
    }

    template<class T>
    inline void waitall(std::vector<PatchDataFieldMpiRequest<T>> &rq_lst){
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
    inline void file_write(MPI_File fh, PatchDataField<T> &p){
        MPI_Status st;

        PatchDataFieldMpiRequest<T> rq (p, current_mode, Send,p.size());

        mpi::file_write(fh, rq.get_mpi_ptr(),  p.size(), get_mpi_type<T>(), &st);

        rq.finalize();
    }
}




