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