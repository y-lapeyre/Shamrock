// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

namespace interact_crit::utils {

    template<class InteractCd,class... Args>
    inline bool interact_cd_cell_patch_domain(const InteractCd & cd, const bool & in_domain, Args ... args){
        bool int_crit;
        if (in_domain) {
            int_crit = InteractCd::interact_cd_cell_patch_outdomain(cd,
                args...
            );
        }else{
            int_crit = InteractCd::interact_cd_cell_patch(cd,
                args...
            );
        } 
        return int_crit;
    }







}
