#pragma once

#include "../aliases.hpp"
#include "../flags.hpp"
#include "../sys/mpi_handler.hpp"
#include "../sys/sycl_mpi_interop.hpp"
#include <mpi.h>
#include <vector>
#include <random>

#include "../utils/sycl_vector_utils.hpp"

namespace patchdata_layout {

    inline u32 nVarpos_s;
    inline u32 nVarpos_d;
    inline u32 nVarU1_s;
    inline u32 nVarU1_d;
    inline u32 nVarU3_s;
    inline u32 nVarU3_d;

    /**
     * @brief should be check if true before communication with patchdata_s
     */
    inline bool layout_synced = false;

    void sync(MPI_Comm comm);

    void set(u32 arg_nVarpos_s, u32 arg_nVarpos_d, u32 arg_nVarU1_s, u32 arg_nVarU1_d, u32 arg_nVarU3_s,
                    u32 arg_nVarU3_d);

    bool is_synced();

} // namespace patchdata_layout



class PatchData {
  public:
    std::vector<f32_3> pos_s;
    std::vector<f64_3> pos_d;
    std::vector<f32>  U1_s;
    std::vector<f64>  U1_d;
    std::vector<f32_3> U3_s;
    std::vector<f64_3> U3_d;
};



void patchdata_isend(PatchData &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm);

PatchData patchdata_irecv( std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm);



inline PatchData patchdata_gen_dummy_data(std::mt19937& eng){

    std::uniform_int_distribution<u64> distu64(1,1000);

    std::uniform_real_distribution<f64> distfd(-1e5,1e5);

    u32 num_part = distu64(eng);

    PatchData d;


    for (u32 i = 0 ; i < num_part; i++) {
        for (u32 ii = 0; ii < patchdata_layout::nVarpos_s; ii ++) {
            d.pos_s.push_back( f32_3{distfd(eng),distfd(eng),distfd(eng)} );
        }
        
        for (u32 ii = 0; ii < patchdata_layout::nVarpos_d; ii ++) {
            d.pos_d.push_back( f64_3{distfd(eng),distfd(eng),distfd(eng)} );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU1_s; ii ++) {
            d.U1_s.push_back( f32(distfd(eng)) );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU1_d; ii ++) {
            d.U1_d.push_back( f64(distfd(eng)) );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU3_s; ii ++) {
            d.U3_s.push_back( f32_3{distfd(eng),distfd(eng),distfd(eng)} );
        }

        for (u32 ii = 0; ii < patchdata_layout::nVarU3_d; ii ++) {
            d.U3_d.push_back( f64_3{distfd(eng),distfd(eng),distfd(eng)} );
        }
    }

    return d;
}

inline bool check_patch_data_match(PatchData& p1, PatchData& p2){
    bool check = true;
    check = check && ( p1.pos_s.size() == p2.pos_s.size());
    check = check && ( p1.pos_d.size() == p2.pos_d.size());
    check = check && ( p1.U1_s.size()  == p2.U1_s.size() );
    check = check && ( p1.U1_d.size()  == p2.U1_d.size() );
    check = check && ( p1.U3_s.size()  == p2.U3_s.size() );
    check = check && ( p1.U3_d.size()  == p2.U3_d.size() );


    for (u32 i = 0; i < p1.pos_s.size(); i ++) {
        check = check && (test_eq3(p1.pos_s[i] , p2.pos_s[i] ));
    }
    
    for (u32 i = 0; i < p1.pos_d.size(); i ++) {
        check = check && (test_eq3(p1.pos_d[i] , p2.pos_d[i] ));
    }

    for (u32 i = 0; i < p1.U1_s.size(); i ++) {
        check = check && (p1.U1_s[i] == p2.U1_s[i] );
    }
    
    for (u32 i = 0; i < p1.U1_d.size(); i ++) {
        check = check && (p1.U1_d[i] == p2.U1_d[i] );
    }

    for (u32 i = 0; i < p1.U3_s.size(); i ++) {
        check = check && (test_eq3(p1.U3_s[i] , p2.U3_s[i] ));
    }
    
    for (u32 i = 0; i < p1.U3_d.size(); i ++) {
        check = check && (test_eq3(p1.U3_d[i] , p2.U3_d[i] ));
    }

    return check;
}