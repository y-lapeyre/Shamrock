#pragma once

#include <vector>

#include "aliases.hpp"

template<class vectype>
class InterfaceHandler{public:

    std::vector<std::tuple<u64,u64,vectype,vectype>> gen_list_interf();

};