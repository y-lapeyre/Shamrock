#pragma once


template<class vec3,class flt>
inline vec3 sph_pressure(
    flt m_b,
    flt rho_a_sq,
    flt rho_b_sq,
    flt P_a,
    flt P_b,
    flt omega_a,
    flt omega_b,
    flt qa_ab,
    flt qb_ab,
    vec3 nabla_Wab_ha,
    vec3 nabla_Wab_hb){

    flt sub_fact_a = rho_a_sq*omega_a;
    flt sub_fact_b = rho_b_sq*omega_b;

    vec3 acc_a = ((P_a + qa_ab)/(sub_fact_a))*nabla_Wab_ha;
    vec3 acc_b = ((P_b + qb_ab)/(sub_fact_b))*nabla_Wab_hb;

    if(sub_fact_a == 0) acc_a = {0,0,0};
    if(sub_fact_b == 0) acc_b = {0,0,0};

    return - m_b*(
          acc_a
        + acc_b
    ); 
}