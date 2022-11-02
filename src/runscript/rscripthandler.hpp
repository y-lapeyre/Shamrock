// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include <string>

class RunScriptHandler {

    wchar_t *program;

    public:

    RunScriptHandler();

    ~RunScriptHandler();

    void run_file(std::string filepath);

    void run_ipython();

};