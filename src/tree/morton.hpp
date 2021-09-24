#pragma once
#include "../aliases.hpp"
#include "../flags.hpp"


namespace morton{

    //21bit number and 2 interleaving bits (for 64bits)
    inline u64 expand_bits_64(u64 x){
        x &= 0x1fffff;
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    }

    inline u64 contract_bits_64(u64 src) {
        //src = src & 0x9249249249249249;
        src = (src | (src >> 2))  & 0x30c30c30c30c30c3;
        src = (src | (src >> 4))  & 0xf00f00f00f00f00f;
        src = (src | (src >> 8))  & 0x00ff0000ff0000ff;
        src = (src | (src >> 16)) & 0xffff00000000ffff;
        src = (src | (src >> 32)) & 0x00000000ffffffff;
        return src;
    }

    //10bit number and 2 interleaving bits (for 32 bits)
    inline u32 expand_bits_32(u32 x){
        x &= 0x3ff;
        x = (x | x << 16) & 0x30000ff;  
        x = (x | x << 8) & 0x300f00f;
        x = (x | x << 4) & 0x30c30c3;
        x = (x | x << 2) & 0x9249249;
        return x;
    }

    inline u32 contract_bits_32(u32 src){
        src = (src | src >> 2) & 0x30C30C3;
        src = (src | src >> 4) & 0xF00F00F;
        src = (src | src >> 8) & 0xFF0000FF;
        src = (src | src >> 16) & 0xFFFF;
        return src;
    }


#if defined(PRECISION_MORTON_DOUBLE)
    inline u64 xyz_to_morton(f_d x, f_d y, f_d z) {

        #if defined(PRECISION_MIXED) || defined(PRECISION_FULL_DOUBLE)
            x = sycl::fmin(sycl::fmax(x * 2097152. , 0.), 2097152.-1.);
            y = sycl::fmin(sycl::fmax(y * 2097152. , 0.), 2097152.-1.);
            z = sycl::fmin(sycl::fmax(z * 2097152. , 0.), 2097152.-1.);
        #else
            x = sycl::fmin(sycl::fmax(x * 2097152.f , 0.f), 2097152.f-1.f);
            y = sycl::fmin(sycl::fmax(y * 2097152.f , 0.f), 2097152.f-1.f);
            z = sycl::fmin(sycl::fmax(z * 2097152.f , 0.f), 2097152.f-1.f);
        #endif

        u64 xx = expand_bits_64((u64)x);
        u64 yy = expand_bits_64((u64)y);
        u64 zz = expand_bits_64((u64)z);
        return xx * 4 + yy * 2 + zz;
    }

    inline u32_3 morton_to_ixyz(u64 morton){
        
        u32_3 pos;
        pos.x() = contract_bits_64((morton & 0x4924924924924924) >> 2);
        pos.y() = contract_bits_64((morton & 0x2492492492492492) >> 1);
        pos.z() = contract_bits_64((morton & 0x1249249249249249) >> 0);
        
        return pos;
    }

    inline u32_3 get_offset(uint clz_){   
        u32_3 mx;
        mx.s0() = 2097152 >> ((clz_ + 1)/3);
        mx.s1() = 2097152 >> ((clz_ - 0)/3);
        mx.s2() = 2097152 >> ((clz_ - 1)/3);
        return mx;
    }

#else

    inline u32 xyz_to_morton(f_d x, f_d y, f_d z) {
        
        #if defined(PRECISION_MIXED) || defined(PRECISION_FULL_DOUBLE)
            x = sycl::fmin(sycl::fmax(x * 1024. , 0.), 1024.-1.);
            y = sycl::fmin(sycl::fmax(y * 1024. , 0.), 1024.-1.);
            z = sycl::fmin(sycl::fmax(z * 1024. , 0.), 1024.-1.);
        #else
            x = sycl::fmin(sycl::fmax(x * 1024.f , 0.f), 1024.f-1.f);
            y = sycl::fmin(sycl::fmax(y * 1024.f , 0.f), 1024.f-1.f);
            z = sycl::fmin(sycl::fmax(z * 1024.f , 0.f), 1024.f-1.f);
        #endif

        u32 xx = expand_bits_32((u32)x);
        u32 yy = expand_bits_32((u32)y);
        u32 zz = expand_bits_32((u32)z);

        return xx * 4 + yy * 2 + zz;
    }

    inline u16_3 morton_to_ixyz(u32 morton){
        
        u16_3 pos;
        pos.s0() = (u16) contract_bits_32((morton & 0x24924924) >> 2);
        pos.s1() = (u16) contract_bits_32((morton & 0x12492492) >> 1);
        pos.s2() = (u16) contract_bits_32((morton & 0x09249249) >> 0);
        
        return pos;
    }

    inline u16_3 get_offset(uint clz_){   
        u16_3 mx;
        mx.s0() = 1024 >> ((clz_ - 0)/3);
        mx.s1() = 1024 >> ((clz_ - 1)/3);
        mx.s2() = 1024 >> ((clz_ - 2)/3);
        return mx;
    }



#endif
    
};
