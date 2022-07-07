// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "runscript/rscripthandler.hpp"
#include "aliases.hpp"
#define PY_SSIZE_T_CLEAN
#include <Python.h>



#include <iostream>
#include "pymodule/pyinitshamrock.hpp"

bool is_already_active = false;

RunScriptHandler::RunScriptHandler(){

    std::cout << "initializing python runscripts" << std::endl;

    if(is_already_active){
        throw shamrock_exc("RunScriptHandler is already active");
    }

    program = Py_DecodeLocale("", NULL);
    if (program == NULL) {
        exit(1);
    }

    std::cout << "initializing shamrock py lib" << std::endl;

    Py_SetProgramName(program);  /* optional but recommended */
    PyImport_AppendInittab("shamrock", &PyInit_shamrock);
    

    is_already_active = true;
}

void RunScriptHandler::run_file(std::string filepath){

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "running pyscript : " << filepath << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    Py_Initialize();
    FILE *file = fopen(filepath.c_str(), "r");
    PyRun_SimpleFileExFlags(file, filepath.c_str(), 0, nullptr);
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "pyscript end" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
}




const char* RUN_IPYTHON = R"(

from IPython import start_ipython
from traitlets.config.loader import Config
import sys

c = Config()

banner ="SHAMROCK Ipython terminal\n" + "Python %s\n"%sys.version.split("\n")[0]

c.TerminalInteractiveShell.banner1 = banner

c.TerminalInteractiveShell.banner2 = """### 
import shamrock
###
"""

start_ipython(config=c)

)";

void RunScriptHandler::run_ipython(){
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "-------------- ipython ---------------------" << std::endl;
    Py_Initialize();
    PyRun_SimpleString(RUN_IPYTHON);
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------\n" << std::endl;
}

RunScriptHandler::~RunScriptHandler(){
    
    PyMem_RawFree(program);

    is_already_active = true;
}