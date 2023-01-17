
#pragma once

#ifdef EXECUTABLE_BUILD
#include <pybind11/embed.h>
#define SHAMROCK_PY_MODULE(name,module) PYBIND11_EMBEDDED_MODULE(name, module)
#endif

#ifdef LIB_BUILD
#include <pybind11/pybind11.h>
#define SHAMROCK_PY_MODULE(name,module) PYBIND11_MODULE(name, module)
#endif


namespace py = pybind11;