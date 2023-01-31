// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patchdata.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief header for PatchData related function and declaration
 * @version 0.1
 * @date 2022-02-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <random>
#include <variant>
#include <vector>

#include "aliases.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "flags.hpp"
#include "patchdata_field.hpp"
#include "shamsys/legacy/mpi_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"

#include "shamrock/patch/PatchDataLayout.hpp"


/**
 * @brief PatchData container class, the layout is described in patchdata_layout
 */
class PatchData {

    void init_fields();


    using var_t = std::variant<
        PatchDataField<f32   >, 
        PatchDataField<f32_2 >, 
        PatchDataField<f32_3 >, 
        PatchDataField<f32_4 >, 
        PatchDataField<f32_8 >, 
        PatchDataField<f32_16>, 
        PatchDataField<f64   >, 
        PatchDataField<f64_2 >, 
        PatchDataField<f64_3 >, 
        PatchDataField<f64_4 >, 
        PatchDataField<f64_8 >, 
        PatchDataField<f64_16>, 
        PatchDataField<u32   >, 
        PatchDataField<u64   >, 
        PatchDataField<u32_3 >, 
        PatchDataField<u64_3 >
        >;

    std::vector<var_t> fields;

    

  public:
    PatchDataLayout & pdl;

    inline PatchData(PatchDataLayout & pdl) : pdl(pdl){
        init_fields();
    }

    template<class Functor>
    inline void for_each_field_any(Functor && func){
        for(auto & f : fields){
            std::visit([&](auto & arg){
                func(arg);
            },f);
        }
    }

    inline PatchData(const PatchData & other) : pdl(other.pdl){

        for(auto & field_var : other.fields){

            std::visit([&](auto & field){

                using base_t = typename std::remove_reference<decltype(field)>::type::Field_type;
                fields.emplace_back(PatchDataField<base_t>(field));

            }, field_var);

        };

    }


    inline PatchData duplicate(){
        const PatchData& current = *this;
        return PatchData(current);
    }

    inline std::unique_ptr<PatchData> duplicate_to_ptr(){
        const PatchData& current = *this;
        return std::make_unique<PatchData>(current);
    }

    PatchData &operator=(const PatchData &other) = delete;
    

    /**
     * @brief extract particle at index pidx and insert it in the provided vectors
     * 
     * @param pidx 
     * @param out_pos_s 
     * @param out_pos_d 
     * @param out_U1_s 
     * @param out_U1_d 
     * @param out_U3_s 
     * @param out_U3_d 
     */
    void extract_element(u32 pidx, PatchData & out_pdat);

    void insert_elements(PatchData & pdat);

    void resize(u32 new_obj_cnt);


    template<class Tvecbox>
    void split_patchdata(PatchData & pd0,PatchData & pd1,PatchData & pd2,PatchData & pd3,PatchData & pd4,PatchData & pd5,PatchData & pd6,PatchData & pd7,
        Tvecbox bmin_p0,Tvecbox bmin_p1,Tvecbox bmin_p2,Tvecbox bmin_p3,Tvecbox bmin_p4,Tvecbox bmin_p5,Tvecbox bmin_p6,Tvecbox bmin_p7,
        Tvecbox bmax_p0,Tvecbox bmax_p1,Tvecbox bmax_p2,Tvecbox bmax_p3,Tvecbox bmax_p4,Tvecbox bmax_p5,Tvecbox bmax_p6,Tvecbox bmax_p7);
    
    void append_subset_to(std::vector<u32> & idxs, PatchData & pdat) const ;
    void append_subset_to(sycl::buffer<u32> & idxs, u32 sz, PatchData & pdat) const ;

    inline u32 get_obj_cnt(){

        bool is_empty = fields.empty();

        if(!is_empty){
            return std::visit([](auto & field){
                return field.get_obj_cnt();
            }, fields[0]);
        }
        
        throw std::runtime_error("this patchdata does not contains any fields");
        
    }

    inline u64 memsize(){
        u64 sum = 0; 

        for(auto & field_var : fields){

            std::visit([&](auto & field){
                sum += field.memsize();
            },field_var);
        }

        return sum;
    }

    inline bool is_empty(){
        return get_obj_cnt() == 0;
    }

    void overwrite(PatchData & pdat, u32 obj_cnt);









    template<class T> PatchDataField<T> & get_field(u32 idx){

        var_t & tmp = fields[idx];

        PatchDataField<T>* pval = std::get_if<PatchDataField<T>>(&tmp);

        if(pval){
            return *pval;
        }

        throw std::runtime_error(
            "the request id is not of correct type\n"
            "   current map is : \n" + pdl.get_description_str() + 
            " this call : " + std::string(__PRETTY_FUNCTION__) + "\n"
            "    arg : idx = " + std::to_string(idx) 
        );
        
    }





    //template<class T> inline std::vector<PatchDataField<T> & > get_field_list(){
    //    std::vector<PatchDataField<T> & > ret;
//
    //    
    //}

    template<class T, class Functor>
    inline void for_each_field(Functor && func){
        for(auto & f : fields){
            PatchDataField<T>* pval = std::get_if<PatchDataField<T>>(&f);

            if(pval){
                func(*pval);
            }
        }
    }

    inline friend bool operator==(const PatchData & p1, const PatchData & p2) { 
        bool check = true;

        if(p1.fields.size() != p2.fields.size()){
            return false;
        }

        for(u32 idx = 0; idx < p1.fields.size(); idx++){

            bool ret = std::visit([&](auto & pf1, auto & pf2) -> bool {

                using t1 = typename std::remove_reference<decltype(pf1)>::type::Field_type;
                using t2 = typename std::remove_reference<decltype(pf2)>::type::Field_type;

                if constexpr (std::is_same<t1, t2>::value){
                    return pf1.check_field_match(pf2);
                }else{  
                    return false;
                }

            }, p1.fields[idx], p2.fields[idx]);


            check = check && ret;
        }


        return check;
    }
    
};


struct PatchDataMpiRequest{
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32   >> mpi_rq_fields_f32;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_2 >> mpi_rq_fields_f32_2;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_3 >> mpi_rq_fields_f32_3;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_4 >> mpi_rq_fields_f32_4;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_8 >> mpi_rq_fields_f32_8;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_16>> mpi_rq_fields_f32_16;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64   >> mpi_rq_fields_f64;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_2 >> mpi_rq_fields_f64_2;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_3 >> mpi_rq_fields_f64_3;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_4 >> mpi_rq_fields_f64_4;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_8 >> mpi_rq_fields_f64_8;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_16>> mpi_rq_fields_f64_16;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<u32   >> mpi_rq_fields_u32;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<u64   >> mpi_rq_fields_u64;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<u32_3   >> mpi_rq_fields_u32_3;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<u64_3   >> mpi_rq_fields_u64_3;

    inline void finalize(){
        for(auto b : mpi_rq_fields_f32   ){b.finalize();}
        for(auto b : mpi_rq_fields_f32_2 ){b.finalize();}
        for(auto b : mpi_rq_fields_f32_3 ){b.finalize();}
        for(auto b : mpi_rq_fields_f32_4 ){b.finalize();}
        for(auto b : mpi_rq_fields_f32_8 ){b.finalize();}
        for(auto b : mpi_rq_fields_f32_16){b.finalize();}
        for(auto b : mpi_rq_fields_f64   ){b.finalize();}
        for(auto b : mpi_rq_fields_f64_2 ){b.finalize();}
        for(auto b : mpi_rq_fields_f64_3 ){b.finalize();}
        for(auto b : mpi_rq_fields_f64_4 ){b.finalize();}
        for(auto b : mpi_rq_fields_f64_8 ){b.finalize();}
        for(auto b : mpi_rq_fields_f64_16){b.finalize();}
        for(auto b : mpi_rq_fields_u32   ){b.finalize();}
        for(auto b : mpi_rq_fields_u64   ){b.finalize();}
        for(auto b : mpi_rq_fields_u32_3   ){b.finalize();}
        for(auto b : mpi_rq_fields_u64_3   ){b.finalize();}
    }

    template<class T> std::vector<patchdata_field::PatchDataFieldMpiRequest<T>> & get_field_list();
    #define X(_arg) template<> inline std::vector<patchdata_field::PatchDataFieldMpiRequest<_arg>> & get_field_list(){return mpi_rq_fields_##_arg ;}
    XMAC_LIST_ENABLED_FIELD
    #undef X
}; 

inline void waitall_pdat_mpi_rq(std::vector<PatchDataMpiRequest> & rq_lst){
    
    std::vector<MPI_Request> rqst;

    auto insertor = [&](auto in){
        std::vector<MPI_Request> rloc = patchdata_field::get_rqs(in);
        rqst.insert(rqst.end(), rloc.begin(), rloc.end());
    };

    for(auto a : rq_lst){
        insertor(a.mpi_rq_fields_f32   );
        insertor(a.mpi_rq_fields_f32_2 );
        insertor(a.mpi_rq_fields_f32_3 );
        insertor(a.mpi_rq_fields_f32_4 );
        insertor(a.mpi_rq_fields_f32_8 );
        insertor(a.mpi_rq_fields_f32_16);
        insertor(a.mpi_rq_fields_f64   );
        insertor(a.mpi_rq_fields_f64_2 );
        insertor(a.mpi_rq_fields_f64_3 );
        insertor(a.mpi_rq_fields_f64_4 );
        insertor(a.mpi_rq_fields_f64_8 );
        insertor(a.mpi_rq_fields_f64_16);
        insertor(a.mpi_rq_fields_u32   );
        insertor(a.mpi_rq_fields_u64   ); 
        insertor(a.mpi_rq_fields_u32_3   );
        insertor(a.mpi_rq_fields_u64_3   );        
    }

    std::vector<MPI_Status> st_lst(rqst.size());
    mpi::waitall(rqst.size(), rqst.data(), st_lst.data());

    for(auto a : rq_lst){
        a.finalize();
    }
}

/**
 * @brief perform a MPI isend with a PatchData object
 *
 * @param p the patchdata to send
 * @param rq_lst reference to the vector of MPI_Request corresponding to the send
 * @param rank_dest rabk to send data to
 * @param tag MPI communication tag
 * @param comm MPI communicator
 */
u64 patchdata_isend(PatchData &p, std::vector<PatchDataMpiRequest> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm);

/**
 * @brief perform a MPI irecv with a PatchData object
 *
 * @param rq_lst reference to the vector of MPI_Request corresponding to the recv
 * @param rank_source rank to receive from
 * @param tag MPI communication tag
 * @param comm  MPI communicator
 * @return the received patchdata (it works but weird because asynchronous)
 */
u64 patchdata_irecv_probe(PatchData &pdat, std::vector<PatchDataMpiRequest> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm);

/**
 * @brief generate dummy patchdata from a mersen twister
 *
 * @param eng the mersen twister
 * @return PatchData the generated PatchData
 */
PatchData patchdata_gen_dummy_data(PatchDataLayout & pdl, std::mt19937 &eng);

/**
 * @brief check if two PatchData content match
 *
 * @param p1
 * @param p2
 * @return true
 * @return false
 */
bool patch_data_check_match(PatchData &p1, PatchData &p2);

