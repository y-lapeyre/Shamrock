#include "../shamrocktest.hpp"

#include <mpi.h>
#include <random>
#include <vector>

#include "../../scheduler/patchdata.hpp"
#include "../../scheduler/mpi_scheduler.hpp"

Test_start("patchdata::", sync_patchdata_layout, -1) {

    if (mpi_handler::world_rank == 0) {
        patchdata_layout::set(1, 8, 4, 6, 2, 1);
    }

    patchdata_layout::sync(MPI_COMM_WORLD);

    Test_assert("sync nVarpos_s",patchdata_layout::nVarpos_s == 1);
    Test_assert("sync nVarpos_d",patchdata_layout::nVarpos_d == 8);
    Test_assert("sync nVarU1_s ",patchdata_layout::nVarU1_s  == 4);
    Test_assert("sync nVarU1_d ",patchdata_layout::nVarU1_d  == 6);
    Test_assert("sync nVarU3_s ",patchdata_layout::nVarU3_s  == 2);
    Test_assert("sync nVarU3_d ",patchdata_layout::nVarU3_d  == 1);

}





template<class T>
bool test_eq3(T a , T b){
    bool eqx = a.x() == b.x();
    bool eqy = a.y() == b.y();
    bool eqz = a.z() == b.z();
    return eqx && eqy && eqz;
}


void check_patch_data_equal(TestResults &__test_result_ref, PatchData& p1, PatchData& p2){

    Test_assert("same_count_pos_s", p1.pos_s.size() == p2.pos_s.size());
    Test_assert("same_count_pos_d", p1.pos_d.size() == p2.pos_d.size());
    Test_assert("same_count_U1_s", p1.U1_s.size() == p2.U1_s.size());
    Test_assert("same_count_U1_d", p1.U1_d.size() == p2.U1_d.size());
    Test_assert("same_count_U3_s", p1.U3_s.size() == p2.U3_s.size());
    Test_assert("same_count_U3_d", p1.U3_d.size() == p2.U3_d.size());




    for (u32 i = 0; i < p1.pos_s.size(); i ++) {
        Test_assert("same value pos_s",test_eq3(p1.pos_s[i] , p2.pos_s[i] ));
    }
    
    for (u32 i = 0; i < p1.pos_d.size(); i ++) {
        Test_assert("same value pos_d",test_eq3(p1.pos_d[i] , p2.pos_d[i] ));
    }

    for (u32 i = 0; i < p1.U1_s.size(); i ++) {
        Test_assert("same value U1_s",p1.U1_s[i] == p2.U1_s[i] );
    }
    
    for (u32 i = 0; i < p1.U1_d.size(); i ++) {
        Test_assert("same value U1_d",p1.U1_d[i] == p2.U1_d[i] );
    }

    for (u32 i = 0; i < p1.U3_s.size(); i ++) {
        Test_assert("same value U3_s",test_eq3(p1.U3_s[i] , p2.U3_s[i] ));
    }
    
    for (u32 i = 0; i < p1.U3_d.size(); i ++) {
        Test_assert("same value U3_d",test_eq3(p1.U3_d[i] , p2.U3_d[i] ));
    }
}


Test_start("patchdata::", send_recv_patchdata, 2){

    std::mt19937 eng(0x1111);  


    if (mpi_handler::world_rank == 0) {
        patchdata_layout::set(1, 8, 4, 6, 2, 1);
    }

    patchdata_layout::sync(MPI_COMM_WORLD);

    create_sycl_mpi_types();



    PatchData d1_check;PatchData d2_check;

    {
        std::uniform_int_distribution<u64> distu64(1,10000);
        std::uniform_real_distribution<f64> distfd(-1e5,1e5);
        u32 num_part = distu64(eng);
        for (u32 i = 0 ; i < num_part; i++) {
            for (u32 ii = 0; ii < patchdata_layout::nVarpos_s; ii ++) {
                d1_check.pos_s.push_back( f3_s{distfd(eng),distfd(eng),distfd(eng)} );
            }
            
            for (u32 ii = 0; ii < patchdata_layout::nVarpos_d; ii ++) {
                d1_check.pos_d.push_back( f3_d{distfd(eng),distfd(eng),distfd(eng)} );
            }

            for (u32 ii = 0; ii < patchdata_layout::nVarU1_s; ii ++) {
                d1_check.U1_s.push_back( f_s(distfd(eng)) );
            }

            for (u32 ii = 0; ii < patchdata_layout::nVarU1_d; ii ++) {
                d1_check.U1_d.push_back( f_d(distfd(eng)) );
            }

            for (u32 ii = 0; ii < patchdata_layout::nVarU3_s; ii ++) {
                d1_check.U3_s.push_back( f3_s{distfd(eng),distfd(eng),distfd(eng)} );
            }

            for (u32 ii = 0; ii < patchdata_layout::nVarU3_d; ii ++) {
                d1_check.U3_d.push_back( f3_d{distfd(eng),distfd(eng),distfd(eng)} );
            }
        }
    }

    {
        std::uniform_int_distribution<u64> distu64(1,10000);
        std::uniform_real_distribution<f64> distfd(-1e5,1e5);
        u32 num_part = distu64(eng);
        for (u32 i = 0 ; i < num_part; i++) {
            for (u32 ii = 0; ii < patchdata_layout::nVarpos_s; ii ++) {
                d2_check.pos_s.push_back( f3_s{distfd(eng),distfd(eng),distfd(eng)} );
            }
            
            for (u32 ii = 0; ii < patchdata_layout::nVarpos_d; ii ++) {
                d2_check.pos_d.push_back( f3_d{distfd(eng),distfd(eng),distfd(eng)} );
            }

            for (u32 ii = 0; ii < patchdata_layout::nVarU1_s; ii ++) {
                d2_check.U1_s.push_back( f_s(distfd(eng)) );
            }

            for (u32 ii = 0; ii < patchdata_layout::nVarU1_d; ii ++) {
                d2_check.U1_d.push_back( f_d(distfd(eng)) );
            }

            for (u32 ii = 0; ii < patchdata_layout::nVarU3_s; ii ++) {
                d2_check.U3_s.push_back( f3_s{distfd(eng),distfd(eng),distfd(eng)} );
            }

            for (u32 ii = 0; ii < patchdata_layout::nVarU3_d; ii ++) {
                d2_check.U3_d.push_back( f3_d{distfd(eng),distfd(eng),distfd(eng)} );
            }
        }
    }


    std::vector<MPI_Request> rq_lst;
    PatchData recv_d;

    if(mpi_handler::world_rank == 0){
        patchdata_isend(d1_check, rq_lst, 1, 0, MPI_COMM_WORLD);
        recv_d = patchdata_irecv(rq_lst, 1, 0, MPI_COMM_WORLD);
    }

    if(mpi_handler::world_rank == 1){
        patchdata_isend(d2_check, rq_lst, 0, 0, MPI_COMM_WORLD);
        recv_d = patchdata_irecv(rq_lst, 0, 0, MPI_COMM_WORLD);
    }

    std::vector<MPI_Status> st_lst(rq_lst.size());
    mpi::waitall(rq_lst.size(), rq_lst.data(), st_lst.data());


    if(mpi_handler::world_rank == 0){
        check_patch_data_equal(__test_result_ref,recv_d, d2_check);
    }

    if(mpi_handler::world_rank == 1){
        check_patch_data_equal(__test_result_ref,recv_d, d1_check);
    }


    free_sycl_mpi_types();
}