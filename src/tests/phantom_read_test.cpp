// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/bytestream.hpp"
#include "shambase/exception.hpp"
#include "shambase/fortran_io.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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
    from_file(shambase::FortranIOFile &phfile, i64 tot_count, std::array<i32, 8> numarray) {
        PhantomBlock block;

        block.tot_count = tot_count;

        for (u32 j = 0; j < numarray[0]; j++) {
            block.table_header_fort_int.push_back(
                PhantomBlockArray<fort_int>::from_file(phfile, block.tot_count));
        }
        for (u32 j = 0; j < numarray[1]; j++) {
            block.table_header_i8.push_back(
                PhantomBlockArray<i8>::from_file(phfile, block.tot_count));
        }
        for (u32 j = 0; j < numarray[2]; j++) {
            block.table_header_i16.push_back(
                PhantomBlockArray<i16>::from_file(phfile, block.tot_count));
        }
        for (u32 j = 0; j < numarray[3]; j++) {
            block.table_header_i32.push_back(
                PhantomBlockArray<i32>::from_file(phfile, block.tot_count));
        }
        for (u32 j = 0; j < numarray[4]; j++) {
            block.table_header_i64.push_back(
                PhantomBlockArray<i64>::from_file(phfile, block.tot_count));
        }
        for (u32 j = 0; j < numarray[5]; j++) {
            block.table_header_fort_real.push_back(
                PhantomBlockArray<fort_real>::from_file(phfile, block.tot_count));
        }
        for (u32 j = 0; j < numarray[6]; j++) {
            block.table_header_f32.push_back(
                PhantomBlockArray<f32>::from_file(phfile, block.tot_count));
        }
        for (u32 j = 0; j < numarray[7]; j++) {
            block.table_header_f64.push_back(
                PhantomBlockArray<f64>::from_file(phfile, block.tot_count));
        }

        return block;
    }
};

struct PhantomDumpData {

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

    static PhantomDumpData from_file(shambase::FortranIOFile &phfile) {
        PhantomDumpData phdump;

        // first line
        //<4 bytes>i1,r1,i2,iversion,i3<4 bytes>
        phfile.read(phdump.i1, phdump.r1, phdump.i2, phdump.iversion, phdump.i3);
        phdump.check_magic_numbers();

        // The second line contains a 100-character file identifier:
        // <4 bytes>fileid<4 bytes>
        phfile.read_fixed_string(phdump.fileid, 100);

        // loop i=1,8
        //    <4 bytes>nvars<4 bytes>
        //    <4 bytes>tags(1:nvars)<4 bytes>
        //    <4 bytes>vals(1:nvals)<4 bytes>
        // end loop
        phdump.table_header_fort_int  = PhantomDumpTableHeader<fort_int>::from_file(phfile);
        phdump.table_header_i8        = PhantomDumpTableHeader<i8>::from_file(phfile);
        phdump.table_header_i16       = PhantomDumpTableHeader<i16>::from_file(phfile);
        phdump.table_header_i32       = PhantomDumpTableHeader<i32>::from_file(phfile);
        phdump.table_header_i64       = PhantomDumpTableHeader<i64>::from_file(phfile);
        phdump.table_header_fort_real = PhantomDumpTableHeader<fort_real>::from_file(phfile);
        phdump.table_header_f32       = PhantomDumpTableHeader<f32>::from_file(phfile);
        phdump.table_header_f64       = PhantomDumpTableHeader<f64>::from_file(phfile);

        int nblocks;
        phfile.read(nblocks);

        std::vector<i64> block_tot_counts;
        std::vector<std::array<i32, 8>> block_numarray;

        for (u32 i = 0; i < nblocks; i++) {

            i64 tot_count;
            std::array<i32, 8> counts;

            phfile.read(tot_count, counts);

            block_tot_counts.push_back(tot_count);
            block_numarray.push_back(counts);
        }
        for (u32 i = 0; i < nblocks; i++) {
            phdump.blocks.push_back(
                PhantomBlock::from_file(phfile, block_tot_counts[i], block_numarray[i]));
        }

        if (!phfile.finished_read()) {
            logger::warn_ln("[PhantomReader]", "some data was not read");
        }

        return phdump;
    }
};

TestStart(Unittest, "phantom-read", pahntomread, 1) {

    std::string fname = "../../exemples/comp-phantom/blast_00472";

    shambase::FortranIOFile phfile = shambase::load_fortran_file(fname);

    i32 fortran_byte;

    PhantomDumpData phdump = PhantomDumpData::from_file(phfile);

    logger::raw_ln(phdump.fileid);
}