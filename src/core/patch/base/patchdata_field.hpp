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
#include <random>
#include <vector>

#include "aliases.hpp"

#include "core/sys/log.hpp"
#include "core/sys/sycl_handler.hpp"
#include "core/sys/sycl_mpi_interop.hpp"
#include "core/utils/string_utils.hpp"
#include "core/utils/sycl_vector_utils.hpp"
#include "core/sys/mpi_handler.hpp"


template<class T>
inline void copydata(T* source, T* dest, u32 cnt){
    logger::debug_alloc_ln("PatchDataField", "copy data src:",source , "dest:",dest, "len=",cnt);
    for (u32 i = 0; i < cnt; i++) {
        dest[i] = source[i];
    }
}


template<class T>
class PatchDataField {

    T* _data = nullptr;
    //std::vector<T> field_data;

    std::string field_name;

    u32 nvar; //number of variable per object
    u32 obj_cnt; // number of contained object

    u32 val_cnt; // nvar*obj_cnt

    u32 capacity;

    constexpr static u32 min_capa = 100;
    constexpr static f32 safe_fact = 1.25;


    void _alloc(){
        _data = new T[capacity];logger::debug_alloc_ln("PatchDataField", "allocate field :",_data , "len =",capacity);
    }

    void _free(){
        if(_data != nullptr) {
            logger::debug_alloc_ln("PatchDataField", "free field :",_data , "len =",capacity);
            delete[] _data;
        }
    }

    public:

    //TODO find a way to add particles easily cf setup require public vector
    

    using Field_type = T;




    inline PatchDataField(std::string name, u32 nvar) : field_name(name) , nvar(nvar){
        obj_cnt = 0;
        val_cnt = 0;

        capacity = 0;
        //_alloc();

    };


    PatchDataField(const PatchDataField& other) {

        field_name  = other.field_name  ;
        nvar        = other.nvar        ; 
        obj_cnt     = other.obj_cnt     ;
        val_cnt     = other.val_cnt     ; 
        capacity    = other.capacity    ;

        //field_data = other.field_data;

        if (capacity != 0) {
            _alloc();
            copydata(other._data,_data, capacity);
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
        logger::debug_alloc_ln("PatchDataField", "free field :",_data , "len =",capacity);
        _free();
    }




    inline T* usm_data(){
        //return field_data.data();
        return _data;
    }

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

    //add overflow check
    inline void resize(u32 new_obj_cnt){

        logger::debug_alloc_ln("PatchDataField", "resize from : ",val_cnt, "to :",new_obj_cnt*nvar);

        u32 new_size = new_obj_cnt*nvar;
        //field_data.resize(new_size);

        //*
        if(capacity == 0){
            capacity = safe_fact*new_size;
            _alloc();
        }else if (new_size > capacity) {
            u32 new_capa = safe_fact*new_size;
            T* new_ptr = new T[new_capa];       logger::debug_alloc_ln("PatchDataField", "allocate : ",new_ptr, "capacity :",new_capa);
            copydata(_data, new_ptr, val_cnt);  
            delete [] _data;                    logger::debug_alloc_ln("PatchDataField", "delete old buf : ",_data);
            _data = new_ptr;
            capacity = new_capa;
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



    inline void insert_element(T v){
        u32 ins_pos = val_cnt;
        expand(1);
        //field_data[ins_pos] = v;
        _data[ins_pos] = v;
    }

    inline void apply_offset(T off){
        for (u32 i = 0; i < val_cnt; i++) {
            _data[i] += off;
        }
        //for(T & v : field_data){
        //    v += off;
        //}
    }


    inline void insert(PatchDataField<T> &f2){

        const u32 idx_st = val_cnt;//field_data.size();
        expand(f2.obj_cnt);

        for (u32 i = 0; i < f2.val_cnt; i++) {
            //field_data[idx_st + i] = f2.field_data[i];
            _data[idx_st + i] = f2._data[i];
        }

    }

    inline void override(sycl::buffer<T> & data){

        if(data.size() !=  val_cnt) throw shamrock_exc("buffer size doesn't match patchdata field size");

        if(val_cnt > 0) {
            auto acc = data.template get_access<sycl::access::mode::read>();

            for(u32 i = 0; i < val_cnt ; i++){
                //field_data[i] = acc[i];
                _data[i] = acc[i];
            }
        }
        
    }

    inline void override(const T val){

        if(val_cnt > 0) {
            for(u32 i = 0; i < val_cnt ; i++){
                //field_data[i] = val;
                _data[i] = val;
            }
        }
        
    }

    // use only if nvar = 1
    template<class Lambdacd>
    inline std::vector<u32> get_elements_with_range(Lambdacd && cd_true, T vmin, T vmax){
        std::vector<u32> idxs;

        for(u32 i = 0; i < val_cnt ; i++){
            if (cd_true(_data[i], vmin, vmax)) {
                idxs.push_back(i);
            }
        }
        
        return idxs;
    }

    inline bool check_field_match(PatchDataField<T> &f2){
        bool match = true;

        match = match && (field_name == f2.field_name);
        match = match && (nvar       == f2.nvar);
        match = match && (obj_cnt    == f2.obj_cnt);
        match = match && (val_cnt    == f2.val_cnt);

        //std::cout << "fieldname : " << field_name << std::endl;
        //std::cout << "val_cnt : " << val_cnt << std::endl;

        for (u32 i = 0; i < val_cnt; i++) {
            //std::cout << i << " " << test_sycl_eq(data()[i],f2.data()[i]) << " " ;
            //print_vec(std::cout, data()[i]);
            //std::cout <<" ";
            //print_vec(std::cout, f2.data()[i]);
            //std::cout <<  std::endl;
            match = match && test_sycl_eq(usm_data()[i],f2.usm_data()[i]);
        }

        return match;
    }

    /**
     * @brief Copy all objects in idxs to pfield
     * 
     * @param idxs 
     * @param pfield 
     */
    inline void append_subset_to(std::vector<u32> & idxs, PatchDataField & pfield){

        if(pfield.nvar != nvar) throw shamrock_exc("field must be similar for extraction");

        const u32 start_enque = pfield.val_cnt;

        const u32 nvar = get_nvar();

        pfield.expand(idxs.size());

        for (u32 i = 0; i < idxs.size(); i++) {

            const u32 idx_extr = idxs[i]*nvar;
            const u32 idx_push = start_enque + i*nvar;

            for(u32 a = 0; a < nvar ; a++){
                pfield.usm_data()[idx_push + a] = usm_data()[idx_extr + a];
            }

        }
    }


    void gen_mock_data(u32 obj_cnt, std::mt19937& eng);



};

namespace patchdata_field {

    template<class T>
    struct PatchDataFieldMpiRequest{
        MPI_Request mpi_rq;

        inline void finalize(){}
    };


    template<class T>
    inline u64 isend( PatchDataField<T> &p, std::vector<PatchDataFieldMpiRequest<T>> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm){
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(p.usm_data(), p.size(), get_mpi_type<T>(), rank_dest, tag, comm, &(rq_lst[rq_lst.size() - 1].mpi_rq));
        return sizeof(T)*p.size();
    }

    template<class T>
    inline u64 irecv(PatchDataField<T> &p, std::vector<PatchDataFieldMpiRequest<T>> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, get_mpi_type<T>(), &cnt);

        u32 len = cnt / p.get_nvar();

        p.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(p.usm_data(), cnt, get_mpi_type<T>(), rank_source, tag, comm, &(rq_lst[rq_lst.size() - 1].mpi_rq));

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
}




