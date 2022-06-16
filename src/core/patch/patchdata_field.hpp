// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include "aliases.hpp"

#include "core/sys/sycl_mpi_interop.hpp"
#include "core/utils/sycl_vector_utils.hpp"
#include "core/sys/mpi_handler.hpp"

template<class T>
class PatchDataField {

    std::vector<T> field_data;

    std::string field_name;

    u32 nvar;
    u32 obj_cnt;

    u32 val_cnt;

    public:

    //TODO find a way to add particles easily cf setup require public vector
    

    using Field_type = T;

    inline PatchDataField(std::string name, u32 nvar) : field_name(name) , nvar(nvar){
        obj_cnt = 0;
        val_cnt = 0;
    };

    inline T* usm_data(){
        return field_data.data();
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

    inline void resize(u32 new_obj_cnt){
        field_data.resize(new_obj_cnt*nvar);
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
        field_data[ins_pos] = v;
    }

    inline void apply_offset(T off){
        for(T & v : field_data){
            v += off;
        }
    }


    inline void insert(PatchDataField<T> &f2){

        const u32 idx_st = field_data.size();
        expand(f2.obj_cnt);

        for (u32 i = 0; i < f2.val_cnt; i++) {
            field_data[idx_st + i] = f2.field_data[i];
        }

    }

    inline void override(sycl::buffer<T> & data){

        if(data.size() !=  val_cnt) throw shamrock_exc("buffer size doesn't match patchdata field size");

        if(val_cnt > 0) {
            auto acc = data.template get_access<sycl::access::mode::read>();

            for(u32 i = 0; i < val_cnt ; i++){
                field_data[i] = acc[i];
            }
        }
        
    }

    inline void override(const T val){

        if(val_cnt > 0) {
            for(u32 i = 0; i < val_cnt ; i++){
                field_data[i] = val;
            }
        }
        
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
    inline u64 isend( PatchDataField<T> &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm){
        rq_lst.resize(rq_lst.size() + 1);
        mpi::isend(p.usm_data(), p.size(), get_mpi_type<T>(), rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
        return sizeof(T)*p.size();
    }

    template<class T>
    inline u64 irecv(PatchDataField<T> &p, std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){
        MPI_Status st;
        i32 cnt;
        int i = mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, get_mpi_type<T>(), &cnt);

        u32 len = cnt / p.get_nvar();

        p.resize(len);

        rq_lst.resize(rq_lst.size() + 1);
        mpi::irecv(p.usm_data(), cnt, get_mpi_type<T>(), rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);

        return sizeof(T)*cnt;
    }
}






template<> inline void PatchDataField<f32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32(distf64(eng));
    }
}

template<> inline void PatchDataField<f32_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_2{distf64(eng),distf64(eng)};
    }
}

template<> inline void PatchDataField<f32_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_3{distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> inline void PatchDataField<f32_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_4{distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> inline void PatchDataField<f32_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_8{distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> inline void PatchDataField<f32_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_16{
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)
            };
    }
}





template<> inline void PatchDataField<f64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64(distf64(eng));
    }
}

template<> inline void PatchDataField<f64_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_2{distf64(eng),distf64(eng)};
    }
}

template<> inline void PatchDataField<f64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_3{distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> inline void PatchDataField<f64_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_4{distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> inline void PatchDataField<f64_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_8{distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> inline void PatchDataField<f64_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_16{
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)
            };
    }
}


template<> inline void PatchDataField<u32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_int_distribution<u32> distu32(1,6000);
    for (auto & a : field_data) {
        a = distu32(eng);
    }
}
template<> inline void PatchDataField<u64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_int_distribution<u64> distu64(1,6000);
    for (auto & a : field_data) {
        a = distu64(eng);
    }
}
