#include "pybindaliases.hpp"



SHAMROCK_PY_MODULE(shamrock,m){
    for(auto fct : static_init_shamrock_pybind){
        fct(m);
    }
}
