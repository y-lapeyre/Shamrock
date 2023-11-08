// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PhantomDump.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/bytestream.hpp"
#include "shambase/exception.hpp"
#include "shambase/fortran_io.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestResult.hpp"
#include <array>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace shammodels::sph {

    template<class T>
    struct PhantomDumpTableHeader {
        std::vector<std::pair<std::string, T>> entries;

        static PhantomDumpTableHeader<T> from_file(shambase::FortranIOFile &phfile);

        void write(shambase::FortranIOFile &phfile);
    };

    template<class T>
    struct PhantomDumpBlockArray {

        std::string tag;
        std::vector<T> vals;

        static PhantomDumpBlockArray from_file(shambase::FortranIOFile &phfile, i64 tot_count);

        void write(shambase::FortranIOFile &phfile, i64 tot_count);
    };

    struct PhantomDumpBlock {
        i64 tot_count;

        using fort_real = f64;
        using fort_int  = int;

        std::vector<PhantomDumpBlockArray<fort_int>> table_header_fort_int;
        std::vector<PhantomDumpBlockArray<i8>> table_header_i8;
        std::vector<PhantomDumpBlockArray<i16>> table_header_i16;
        std::vector<PhantomDumpBlockArray<i32>> table_header_i32;
        std::vector<PhantomDumpBlockArray<i64>> table_header_i64;
        std::vector<PhantomDumpBlockArray<fort_real>> table_header_fort_real;
        std::vector<PhantomDumpBlockArray<f32>> table_header_f32;
        std::vector<PhantomDumpBlockArray<f64>> table_header_f64;

        static PhantomDumpBlock
        from_file(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);

        void write(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);
    };

    struct PhantomDump {

        using fort_real = f64;
        using fort_int  = int;

        fort_int i1, i2, iversion, i3;
        fort_real r1;
        std::string fileid;

        void check_magic_numbers() {
            if (i1 != 60769) {
                shambase::throw_with_loc<std::runtime_error>("");
            }
            if (i2 != 60878) {
                shambase::throw_with_loc<std::runtime_error>("");
            }
            if (i3 != 690706) {
                shambase::throw_with_loc<std::runtime_error>("");
            }
            if (r1 != i2) {
                shambase::throw_with_loc<std::runtime_error>("");
            }
        }

        PhantomDumpTableHeader<fort_int> table_header_fort_int;
        PhantomDumpTableHeader<i8> table_header_i8;
        PhantomDumpTableHeader<i16> table_header_i16;
        PhantomDumpTableHeader<i32> table_header_i32;
        PhantomDumpTableHeader<i64> table_header_i64;
        PhantomDumpTableHeader<fort_real> table_header_fort_real;
        PhantomDumpTableHeader<f32> table_header_f32;
        PhantomDumpTableHeader<f64> table_header_f64;

        std::vector<PhantomDumpBlock> blocks;

        shambase::FortranIOFile gen_file();

        static PhantomDump from_file(shambase::FortranIOFile &phfile);
    };
} // namespace shammodels::sph