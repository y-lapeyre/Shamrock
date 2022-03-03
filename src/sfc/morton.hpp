/**
 * @file morton.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Morton curve implementation
 * @version 0.1
 * @date 2022-03-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once

#include "aliases.hpp"
#include "sfc/bmi.hpp"

template<class morton_prec,class fp_prec,class int_repr>
class Morton3D{public:

    morton_prec coord_to_morton(fp_prec x,fp_prec y,fp_prec z);

    int_repr morton_to_ipos(morton_prec morton);
    int_repr get_offset(u32 clz_);

};

template<>
class Morton3D<u64,f64,u32_3>{public:
    inline u64 coord_to_morton(f64 x,f64 y,f64 z){
        x = sycl::fmin(sycl::fmax(x * 2097152. , 0.), 2097152.-1.);
        y = sycl::fmin(sycl::fmax(y * 2097152. , 0.), 2097152.-1.);
        z = sycl::fmin(sycl::fmax(z * 2097152. , 0.), 2097152.-1.);

        u64 xx = bmi::expand_bits<u64,2>((u64)x);
        u64 yy = bmi::expand_bits<u64,2>((u64)y);
        u64 zz = bmi::expand_bits<u64,2>((u64)z);
        return xx * 4 + yy * 2 + zz;
    }

    inline u32_3 morton_to_ixyz(u64 morton){
        
        u32_3 pos;
        pos.x() = bmi::contract_bits<u64, 2>((morton & 0x4924924924924924) >> 2);
        pos.y() = bmi::contract_bits<u64, 2>((morton & 0x2492492492492492) >> 1);
        pos.z() = bmi::contract_bits<u64, 2>((morton & 0x1249249249249249) >> 0);
        
        return pos;
    }

    inline u32_3 get_offset(uint clz_){   
        u32_3 mx;
        mx.s0() = 2097152 >> ((clz_ + 1)/3);
        mx.s1() = 2097152 >> ((clz_ - 0)/3);
        mx.s2() = 2097152 >> ((clz_ - 1)/3);
        return mx;
    }

};


template<>
class Morton3D<u64,f32,u32_3>{public:
    inline u64 coord_to_morton(f32 x,f32 y,f32 z){
        x = sycl::fmin(sycl::fmax(x * 2097152.f , 0.f), 2097152.f-1.f);
        y = sycl::fmin(sycl::fmax(y * 2097152.f , 0.f), 2097152.f-1.f);
        z = sycl::fmin(sycl::fmax(z * 2097152.f , 0.f), 2097152.f-1.f);

        u64 xx = bmi::expand_bits<u64,2>((u64)x);
        u64 yy = bmi::expand_bits<u64,2>((u64)y);
        u64 zz = bmi::expand_bits<u64,2>((u64)z);
        return xx * 4 + yy * 2 + zz;
    }

    inline u32_3 morton_to_ixyz(u64 morton){
        
        u32_3 pos;
        pos.x() = bmi::contract_bits<u64, 2>((morton & 0x4924924924924924) >> 2);
        pos.y() = bmi::contract_bits<u64, 2>((morton & 0x2492492492492492) >> 1);
        pos.z() = bmi::contract_bits<u64, 2>((morton & 0x1249249249249249) >> 0);
        
        return pos;
    }

    inline u32_3 get_offset(uint clz_){   
        u32_3 mx;
        mx.s0() = 2097152 >> ((clz_ + 1)/3);
        mx.s1() = 2097152 >> ((clz_ - 0)/3);
        mx.s2() = 2097152 >> ((clz_ - 1)/3);
        return mx;
    }
};



template<>
class Morton3D<u32,f64,u16_3>{public:
    inline u32 coord_to_morton(f64 x,f64 y,f64 z){
        x = sycl::fmin(sycl::fmax(x * 1024. , 0.), 1024.-1.);
        y = sycl::fmin(sycl::fmax(y * 1024. , 0.), 1024.-1.);
        z = sycl::fmin(sycl::fmax(z * 1024. , 0.), 1024.-1.);

        u32 xx = bmi::expand_bits<u32,2>((u32)x);
        u32 yy = bmi::expand_bits<u32,2>((u32)y);
        u32 zz = bmi::expand_bits<u32,2>((u32)z);
        return xx * 4 + yy * 2 + zz;
    }

    inline u16_3 morton_to_ipos(u32 morton){
        
        u16_3 pos;
        pos.s0() = (u16) bmi::contract_bits<u32,2>((morton & 0x24924924) >> 2);
        pos.s1() = (u16) bmi::contract_bits<u32,2>((morton & 0x12492492) >> 1);
        pos.s2() = (u16) bmi::contract_bits<u32,2>((morton & 0x09249249) >> 0);
        
        return pos;
    }

    inline u16_3 get_offset(uint clz_){   
        u16_3 mx;
        mx.s0() = 1024 >> ((clz_ - 0)/3);
        mx.s1() = 1024 >> ((clz_ - 1)/3);
        mx.s2() = 1024 >> ((clz_ - 2)/3);
        return mx;
    }
};


template<>
class Morton3D<u32,f32,u16_3>{public:
    inline u32 coord_to_morton(f32 x,f32 y,f32 z){
        x = sycl::fmin(sycl::fmax(x * 1024.f , 0.f), 1024.f-1.f);
        y = sycl::fmin(sycl::fmax(y * 1024.f , 0.f), 1024.f-1.f);
        z = sycl::fmin(sycl::fmax(z * 1024.f , 0.f), 1024.f-1.f);

        u32 xx = bmi::expand_bits<u32,2>((u32)x);
        u32 yy = bmi::expand_bits<u32,2>((u32)y);
        u32 zz = bmi::expand_bits<u32,2>((u32)z);
        return xx * 4 + yy * 2 + zz;
    }


    inline u16_3 morton_to_ipos(u32 morton){
        
        u16_3 pos;
        pos.s0() = (u16) bmi::contract_bits<u32,2>((morton & 0x24924924) >> 2);
        pos.s1() = (u16) bmi::contract_bits<u32,2>((morton & 0x12492492) >> 1);
        pos.s2() = (u16) bmi::contract_bits<u32,2>((morton & 0x09249249) >> 0);
        
        return pos;
    }

    inline u16_3 get_offset(uint clz_){   
        u16_3 mx;
        mx.s0() = 1024 >> ((clz_ - 0)/3);
        mx.s1() = 1024 >> ((clz_ - 1)/3);
        mx.s2() = 1024 >> ((clz_ - 2)/3);
        return mx;
    }
};
