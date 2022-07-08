// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "patchdata_field.hpp"
#include "core/patch/base/enabled_fields.hpp"






















#define X(a) template class PatchDataField<a>;
XMAC_LIST_ENABLED_FIELD
#undef X





template<> void PatchDataField<f32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32(distf64(eng));
    }
}

template<> void PatchDataField<f32_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_2{distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f32_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_3{distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f32_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_4{distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f32_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_8{distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f32_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f32_16{
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)
            };
    }
}





template<> void PatchDataField<f64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64(distf64(eng));
    }
}

template<> void PatchDataField<f64_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_2{distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_3{distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f64_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_4{distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f64_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_8{distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f64_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (auto & a : field_data) {
        a = f64_16{
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)
            };
    }
}


template<> void PatchDataField<u32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_int_distribution<u32> distu32(1,6000);
    for (auto & a : field_data) {
        a = distu32(eng);
    }
}
template<> void PatchDataField<u64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_int_distribution<u64> distu64(1,6000);
    for (auto & a : field_data) {
        a = distu64(eng);
    }
}
