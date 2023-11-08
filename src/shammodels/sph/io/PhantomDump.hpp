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

        static PhantomDumpTableHeader<T> from_file(shambase::FortranIOFile &phfile) {
            PhantomDumpTableHeader<T> tmp;

            int nvars;

            phfile.read(nvars);

            if (nvars == 0) {
                return tmp;
            }

            std::vector<std::string> tags;
            phfile.read_string_array(tags, 16, nvars);

            std::vector<T> vals;
            phfile.read_val_array(vals, nvars);

            for (u32 i = 0; i < nvars; i++) {
                tmp.entries.push_back({tags[i], vals[i]});
            }

            return tmp;
        }

        inline void write(shambase::FortranIOFile &phfile) {
            int nvars = entries.size();
            phfile.write(nvars);

            if (nvars == 0) {
                return;
            }

            std::vector<std::string> tags;
            std::vector<T> vals;
            for (u32 i = 0; i < nvars; i++) {
                auto [a, b] = entries[i];
                tags.push_back(a);
                vals.push_back(b);
            }

            phfile.write_string_array(tags, 16, nvars);
            phfile.write_val_array(vals, nvars);
        }
    };

    template<class T>
    struct PhantomBlockArray {

        std::string tag;
        std::vector<T> vals;

        static PhantomBlockArray from_file(shambase::FortranIOFile &phfile, i64 tot_count) {
            PhantomBlockArray tmp;
            phfile.read_fixed_string(tmp.tag, 16);
            phfile.read_val_array(tmp.vals, tot_count);
            return tmp;
        }

        inline void write(shambase::FortranIOFile &phfile, i64 tot_count) {
            phfile.write_fixed_string(tag, 16);
            phfile.write_val_array(vals, tot_count);
        }
    };

    struct PhantomBlock {
        i64 tot_count;

        using fort_real = f64;
        using fort_int  = int;

        std::vector<PhantomBlockArray<fort_int>> table_header_fort_int;
        std::vector<PhantomBlockArray<i8>> table_header_i8;
        std::vector<PhantomBlockArray<i16>> table_header_i16;
        std::vector<PhantomBlockArray<i32>> table_header_i32;
        std::vector<PhantomBlockArray<i64>> table_header_i64;
        std::vector<PhantomBlockArray<fort_real>> table_header_fort_real;
        std::vector<PhantomBlockArray<f32>> table_header_f32;
        std::vector<PhantomBlockArray<f64>> table_header_f64;

        static PhantomBlock
        from_file(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray);

        void
        write(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray) ;
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

        std::vector<PhantomBlock> blocks;

        shambase::FortranIOFile gen_file();

        static PhantomDump from_file(shambase::FortranIOFile &phfile);
    };
} // namespace shammodels::sph