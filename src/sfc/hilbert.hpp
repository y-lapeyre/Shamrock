/**
 * @file hilbertsfc.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief hilbert curve implementation from killing J., 2004
 *
 * modified from :    
 * Programming the Hilbert curve     
 * killing J., 2004, AIPC, 707, 381. doi:10.1063/1.1751381     
 *
 * @version 1.0
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include "aliases.hpp"
#include "bmi.hpp"

// modified from :
// Programming the Hilbert curve
// killing J., 2004, AIPC, 707, 381. doi:10.1063/1.1751381

constexpr u64 hilbert_box21_sz = 2097152 - 1;

template <int b> inline u64 compute_hilbert_index(u64 x, u64 y, u64 z) {

    const int n = 3;
    u64 X[3]    = {x, y, z};

    u64 M = 1 << (b - 1), P, Q, t;
    int i;
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


    X[0] = bmi::expand_bits_64(X[0]) << 2;
    X[1] = bmi::expand_bits_64(X[1]) << 1;
    X[2] = bmi::expand_bits_64(X[2]);

    return X[0] + X[1] + X[2];
}