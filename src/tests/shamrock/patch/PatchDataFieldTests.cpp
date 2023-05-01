// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/memory/serialize.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/patch/PatchDataField::serialize", testpatchdatafieldserialize, 1){
    
    u32 len = 1000;
    u32 nvar = 2;
    std::string name = "testfield";
    PatchDataField<u32_3> field = PatchDataField<u32_3>::mock_field(0x111, len, name, nvar);

    shamalgs::SerializeHelper ser;

    ser.allocate(field.serialize_buf_byte_size());
    field.serialize_buf(ser);


    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(std::move(recov));

        PatchDataField<u32_3> buf2 = PatchDataField<u32_3>::deserialize_buf(ser2, name, nvar);

        shamtest::asserts().assert_bool("input match out", field.check_field_match(buf2));
    }

}