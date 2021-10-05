
#include "../unit_test_handler.hpp"
#include "../../tree/patch.hpp"
#include <vector>

inline void run_tests_patch(){

    if(unit_test::test_start("tree/patch.hpp (Patch data (se/dese)rialization)", false)){

        PatchData data_clean;

        data_clean.r.push_back({0,0,0});
        data_clean.r.push_back({0,0.1,0});
        data_clean.r.push_back({0,0,0});
        data_clean.r.push_back({0.3,0,0});
        data_clean.r.push_back({0,0,0.9});
        data_clean.r.push_back({0.7,0.8,0.9});

        data_clean.obj_cnt = data_clean.r.size();

        std::vector<u8> data_ser = data_clean.serialize();

        printf("%zu\n",data_ser.size());

        printf("%d\n",* ((u32*)data_ser.data()));

        
        PatchData data_tran(data_ser);

        unit_test::test_assert("same obj count", data_clean.obj_cnt, data_tran.obj_cnt);

        for(u32 i = 0; i < data_clean.obj_cnt; i++){
            bool same = 
                (data_clean.r[i].x() == data_tran.r[i].x()) &&
                (data_clean.r[i].y() == data_tran.r[i].y()) &&
                (data_clean.r[i].z() == data_tran.r[i].z())
            ;
            unit_test::test_assert("same r", same);
        }
        

    }unit_test::test_end();




    if(unit_test::test_start("tree/patch.hpp (Sparse collective communication)", true)){

        printf("test patch\n");

    }unit_test::test_end();

}