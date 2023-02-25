// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once
/**
 * @file patchtree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Patch tree for the mpi side of the code
 * @version 0.1
 * @date 2022-02-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */



#include <array>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>

#include "aliases.hpp"
#include "shamrock/patch/Patch.hpp"

#include "shamrock/legacy/utils/geometry_utils.hpp"

#include "shamrock/scheduler/PatchTree.hpp"

