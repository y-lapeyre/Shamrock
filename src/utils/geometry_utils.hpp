#pragma once

#include "../aliases.hpp"
#include "CL/sycl/builtins.hpp"

namespace BBAA {


    template<class VecType> bool iscellb_inside_a(VecType pos_min_cella,VecType pos_max_cella,VecType pos_min_cellb,VecType pos_max_cellb);
    
    template<> inline bool iscellb_inside_a<u32_3>(u32_3 pos_min_cella,u32_3 pos_max_cella,u32_3 pos_min_cellb,u32_3 pos_max_cellb) {
        return (
                (pos_min_cella.x() <= pos_min_cellb.x()) && (pos_min_cellb.x() < pos_max_cellb.x()) && (pos_max_cellb.x() <= pos_max_cella.x()) &&
                (pos_min_cella.y() <= pos_min_cellb.y()) && (pos_min_cellb.y() < pos_max_cellb.y()) && (pos_max_cellb.y() <= pos_max_cella.y()) &&
                (pos_min_cella.z() <= pos_min_cellb.z()) && (pos_min_cellb.z() < pos_max_cellb.z()) && (pos_max_cellb.z() <= pos_max_cella.z()) 
            );
    }

    template<> inline bool iscellb_inside_a<f32_3>(f32_3 pos_min_cella,f32_3 pos_max_cella,f32_3 pos_min_cellb,f32_3 pos_max_cellb) {
        return (
                (pos_min_cella.x() <= pos_min_cellb.x()) && (pos_min_cellb.x() < pos_max_cellb.x()) && (pos_max_cellb.x() <= pos_max_cella.x()) &&
                (pos_min_cella.y() <= pos_min_cellb.y()) && (pos_min_cellb.y() < pos_max_cellb.y()) && (pos_max_cellb.y() <= pos_max_cella.y()) &&
                (pos_min_cella.z() <= pos_min_cellb.z()) && (pos_min_cellb.z() < pos_max_cellb.z()) && (pos_max_cellb.z() <= pos_max_cella.z()) 
            );
    }






    template<class VecType> bool cella_neigh_b(VecType pos_min_cella,VecType pos_max_cella,VecType pos_min_cellb,VecType pos_max_cellb);

    template<> inline bool cella_neigh_b<f32_3>(f32_3 pos_min_cella,f32_3 pos_max_cella,f32_3 pos_min_cellb,f32_3 pos_max_cellb){
        return (
                (sycl::fmax( pos_min_cella.x(), pos_min_cellb.x()) <= sycl::fmin(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::fmax( pos_min_cella.y(), pos_min_cellb.y()) <= sycl::fmin(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::fmax( pos_min_cella.z(), pos_min_cellb.z()) <= sycl::fmin(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }

    template<> inline bool cella_neigh_b<f64_3>(f64_3 pos_min_cella,f64_3 pos_max_cella,f64_3 pos_min_cellb,f64_3 pos_max_cellb){
        return (
                (sycl::fmax( pos_min_cella.x(), pos_min_cellb.x()) <= sycl::fmin(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::fmax( pos_min_cella.y(), pos_min_cellb.y()) <= sycl::fmin(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::fmax( pos_min_cella.z(), pos_min_cellb.z()) <= sycl::fmin(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }




    template<class VecType> bool intersect_not_null_cella_b(VecType pos_min_cella,VecType pos_max_cella,VecType pos_min_cellb,VecType pos_max_cellb);

    template<> inline bool intersect_not_null_cella_b<f64_3>(f64_3 pos_min_cella,f64_3 pos_max_cella,f64_3 pos_min_cellb,f64_3 pos_max_cellb){
        return (
                (sycl::fmax( pos_min_cella.x(), pos_min_cellb.x()) < sycl::fmin(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::fmax( pos_min_cella.y(), pos_min_cellb.y()) < sycl::fmin(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::fmax( pos_min_cella.z(), pos_min_cellb.z()) < sycl::fmin(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }

    template<> inline bool intersect_not_null_cella_b<u64_3>(u64_3 pos_min_cella,u64_3 pos_max_cella,u64_3 pos_min_cellb,u64_3 pos_max_cellb){
        return (
                (sycl::max( pos_min_cella.x(), pos_min_cellb.x()) < sycl::min(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::max( pos_min_cella.y(), pos_min_cellb.y()) < sycl::min(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::max( pos_min_cella.z(), pos_min_cellb.z()) < sycl::min(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }

    template<> inline bool intersect_not_null_cella_b<u32_3>(u32_3 pos_min_cella,u32_3 pos_max_cella,u32_3 pos_min_cellb,u32_3 pos_max_cellb){
        return (
                (sycl::max( pos_min_cella.x(), pos_min_cellb.x()) < sycl::min(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::max( pos_min_cella.y(), pos_min_cellb.y()) < sycl::min(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::max( pos_min_cella.z(), pos_min_cellb.z()) < sycl::min(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }

}