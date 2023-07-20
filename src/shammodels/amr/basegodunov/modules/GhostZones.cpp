// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/basegodunov/modules/GhostZones.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec,TgridVec>::build_ghost_cache(){
    
}