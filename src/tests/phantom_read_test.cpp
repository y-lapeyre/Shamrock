// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/bytestream.hpp"
#include "shambase/exception.hpp"
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


void check_fortran_byte(std::basic_stringstream<byte> & buffer, i32 & fortran_byte){
    i32 new_check = 0;

    shambase::stream_read(buffer, new_check); 
    logger::raw_ln("check f-byte",fortran_byte, new_check);

    if(new_check != fortran_byte){
        throw shambase::throw_with_loc<std::runtime_error>("fortran 4 bytes invalid");
    }

    fortran_byte = new_check;
}







template<class T>
struct PhantomDumpTableHeader{
    std::vector<std::pair<std::string, T>> entries;
};



template<class T>
struct PhantomBlockArray{

};

struct PhantomBlock{

};

struct PhantomDumpData{

    using fort_real = f64;
    using fort_int = int;

    fort_int i1,i2,iversion,i3;
    fort_real r1;
    std::string fileid;

    PhantomDumpTableHeader<fort_int> table_header_fort_int;
    PhantomDumpTableHeader<i8 > table_header_i8 ;
    PhantomDumpTableHeader<i16> table_header_i16;
    PhantomDumpTableHeader<i32> table_header_i32;
    PhantomDumpTableHeader<i64> table_header_i64;
    PhantomDumpTableHeader<fort_real> table_header_fort_real;
    PhantomDumpTableHeader<f32> table_header_f32;
    PhantomDumpTableHeader<f64> table_header_f64;

    void check_magic_numbers(){
        if(i1 != 60769){
        shambase::throw_with_loc<std::runtime_error>("");
        }
        if(i2 != 60878){
            shambase::throw_with_loc<std::runtime_error>("");
        }
        if(i3 != 690706){
            shambase::throw_with_loc<std::runtime_error>("");
        }
        if(r1 != i2){
            shambase::throw_with_loc<std::runtime_error>("");
        }
    }

};


template<class T>
void read_global_header(std::basic_stringstream<byte> & buffer, PhantomDumpTableHeader<T> & table){

    logger::raw_ln(__PRETTY_FUNCTION__);

    i32 nvars;

    i32 fortran_byte;

    shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);
    shambase::stream_read(buffer, nvars);
    check_fortran_byte(buffer, fortran_byte);

    logger::raw_ln("nvars", nvars);

    if(nvars == 0){
        return;
    }

    std::vector<std::string> tags;

    shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);

    for(u32 i = 0; i < nvars; i++){

        std::string tag;
        tag.resize(16);

        buffer.read(reinterpret_cast<byte *>(tag.data()), 16 * sizeof(char));

        tags.push_back(tag);
        logger::raw_ln("tag = ",tag);
    }
    check_fortran_byte(buffer, fortran_byte);



    std::vector<T>  vals;
    shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);

    for(u32 i = 0; i < nvars; i++){

        T val;

        shambase::stream_read(buffer, val);

        vals.push_back(val);
        logger::raw_ln("val = ",val);
    }
    check_fortran_byte(buffer, fortran_byte);

    for (u32 i = 0; i < nvars; i ++) {
        table.entries.push_back({
            tags[i],
            vals[i]
        });
    }

}




template<class T> 
void read_phantom_block_array(std::basic_stringstream<byte> & buffer, PhantomBlockArray<T> & table, i64 totcount){

    i32 fortran_byte;

    
    std::vector<std::string> tags;



    std::string tag;
    tag.resize(16);

    shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);
    buffer.read(reinterpret_cast<byte *>(tag.data()), 16 * sizeof(char));
check_fortran_byte(buffer, fortran_byte);

    tags.push_back(tag);
    logger::raw_ln("tag = ",tag);
    
    


    std::vector<T> vals;
    vals.resize(totcount);


    shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);
    buffer.read(reinterpret_cast<byte *>(vals.data()), totcount * sizeof(T));
    check_fortran_byte(buffer, fortran_byte);
}

TestStart(Unittest, "phantom-read", pahntomread, 1){

    std::string fname = "../../exemples/comp-phantom/blast_00472";

    std::ifstream in_f (fname, std::ios::binary);

    std::basic_stringstream<byte> buffer;
    if(in_f){
        buffer << in_f.rdbuf();
        in_f.close();
    }else {
    shambase::throw_unimplemented();
    }
    buffer.seekg(0); // rewind

    i32 fortran_byte;


    PhantomDumpData phdump;

    //first line
    //<4 bytes>i1,r1,i2,iversion,i3<4 bytes>

    shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);


    shambase::stream_read(buffer, phdump.i1);
    shambase::stream_read(buffer, phdump.r1);
    shambase::stream_read(buffer, phdump.i2);
    shambase::stream_read(buffer, phdump.iversion);
    shambase::stream_read(buffer, phdump.i3);
    phdump.check_magic_numbers();

    check_fortran_byte(buffer, fortran_byte);

    // The second line contains a 100-character file identifier:
    // <4 bytes>fileid<4 bytes>

    phdump.fileid.resize(100);
    shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);
    buffer.read(reinterpret_cast<byte *>(phdump.fileid.data()), 100 * sizeof(char));
    check_fortran_byte(buffer, fortran_byte);


    logger::raw_ln(phdump.fileid);


    //loop i=1,8
    //   <4 bytes>nvars<4 bytes>
    //   <4 bytes>tags(1:nvars)<4 bytes>
    //   <4 bytes>vals(1:nvals)<4 bytes>
    //end loop
    read_global_header<int>(buffer, phdump.table_header_fort_int);
    read_global_header<i8>(buffer, phdump.table_header_i8);
    read_global_header<i16>(buffer, phdump.table_header_i16);
    read_global_header<i32>(buffer, phdump.table_header_i32);
    read_global_header<i64>(buffer, phdump.table_header_i64);
    read_global_header<f64>(buffer, phdump.table_header_fort_real);
    read_global_header<f32>(buffer, phdump.table_header_f32);
    read_global_header<f64>(buffer, phdump.table_header_f64);


    int nblocks;
    shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);
    shambase::stream_read(buffer, nblocks);
    check_fortran_byte(buffer, fortran_byte);

    logger::raw_ln(nblocks);

    std::vector<i64> block_tot_counts;
    std::vector<std::array<i32, 8>> block_numarray;

    for(u32 i = 0; i < nblocks; i++){

        i64 tot_count;
        std::array<i32, 8> counts;

        shambase::stream_read(buffer, fortran_byte); logger::raw_ln("f-byte", fortran_byte);
        shambase::stream_read(buffer, tot_count);
        buffer.read(reinterpret_cast<byte *>(counts.data()), 8 * sizeof(i32));
        check_fortran_byte(buffer, fortran_byte);
        
        block_tot_counts.push_back(tot_count);
        block_numarray.push_back(counts);
    }

    
    for(u32 i = 0; i < nblocks; i++){


        i64 tot_count = block_tot_counts[i];
        std::array<i32, 8> numarray = block_numarray[i];
       
        for(u32 j = 0; j < numarray[0] ; j++){
            PhantomBlockArray<int> tmp;
            read_phantom_block_array(buffer, tmp, tot_count);
        }
        for(u32 j = 0; j < numarray[1] ; j++){
            PhantomBlockArray<i8> tmp;
            read_phantom_block_array(buffer, tmp, tot_count);
        }
        for(u32 j = 0; j < numarray[2] ; j++){
            PhantomBlockArray<i16> tmp;
            read_phantom_block_array(buffer, tmp, tot_count);
        }
        for(u32 j = 0; j < numarray[3] ; j++){
            PhantomBlockArray<i32> tmp;
            read_phantom_block_array(buffer, tmp, tot_count);
        }
        for(u32 j = 0; j < numarray[4] ; j++){
            PhantomBlockArray<i64> tmp;
            read_phantom_block_array(buffer, tmp, tot_count);
        }
        for(u32 j = 0; j < numarray[5] ; j++){
            PhantomBlockArray<f64> tmp;
            read_phantom_block_array(buffer, tmp, tot_count);
        }
        for(u32 j = 0; j < numarray[6] ; j++){
            PhantomBlockArray<f32> tmp;
            read_phantom_block_array(buffer, tmp, tot_count);
        }
        for(u32 j = 0; j < numarray[7] ; j++){
            PhantomBlockArray<f64> tmp;
            read_phantom_block_array(buffer, tmp, tot_count);
        }
        


    }




}