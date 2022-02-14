#pragma once

#include "../aliases.hpp"

// modified from : 
// Programming the Hilbert curve 
// killing J., 2004, AIPC, 707, 381. doi:10.1063/1.1751381


inline u64 expand_bits_64b(u64 x){
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

constexpr u64 hilbert_box21_sz = 2097152-1;

template<int b>
inline u64 compute_hilbert_index(u64 x, u64 y, u64 z){

    const int n = 3;
    u64 X[3] = {x,y,z};

    u64 M = 1 << (b - 1), P, Q, t;
    int    i;
    // Inverse undo
    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        for (i = 0; i < n; i++)
            if (X[i] & Q)
                X[0] ^= P; // invert
            else {
                t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
    } // exchange
    
    // Gray encode
    for (i = 1; i < n; i++)
        X[i] ^= X[i - 1];
    t = 0;
    for (Q = M; Q > 1; Q >>= 1)
        if (X[n - 1] & Q)
            t ^= Q - 1;
    for (i = 0; i < n; i++)
        X[i] ^= t; 

    /*
    for(int n = 15; n >=0 ; n-- ){
        std::cout << (X[0] >> n & 1) << " ";
    }std::cout << std::endl;
    for(int n = 15; n >=0 ; n-- ){
        std::cout << (X[1] >> n & 1) << " ";
    }std::cout << std::endl;
    for(int n = 15; n >=0 ; n-- ){
        std::cout << (X[2] >> n & 1) << " ";
    }std::cout << std::endl;

    std::cout << std::endl;
    */

    X[0] = expand_bits_64b(X[0])<<2;
    X[1] = expand_bits_64b(X[1])<<1;
    X[2] = expand_bits_64b(X[2]);

    /*
    for(int n = 15; n >=0 ; n-- ){
        std::cout << (X[0] >> n & 1) << " ";
    }std::cout << std::endl;
    for(int n = 15; n >=0 ; n-- ){
        std::cout << (X[1] >> n & 1) << " ";
    }std::cout << std::endl;
    for(int n = 15; n >=0 ; n-- ){
        std::cout << (X[2] >> n & 1) << " ";
    }std::cout << std::endl;
    */


    return X[0] + X[1] + X[2];
}