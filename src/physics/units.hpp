#include "aliases.hpp"

namespace units {

namespace SI_cst{

    /*
    * lenghts units
    */
    constexpr f64 m = 1;//(m)
    constexpr f64 km = 1000;//(m)
    constexpr f64 nm = 1e-9;//(m)
    constexpr f64 au = 149597870700; //(m)
    constexpr f64 ly = 9460730472580800; //(m)
    constexpr f64 pc = 3.0857e16;//(m)

    /*
    * time units
    */
    constexpr f64 s = 1;
    constexpr f64 mn = 60;
    constexpr f64 hr = 60*mn;
    constexpr f64 dy = 24*hr;
    constexpr f64 yr = 31557600;


    /*
    * mass units
    */
    constexpr f64 earth_mass = 5.9722e24; //(kg)
    constexpr f64 jupiter_mass = 1.898e27; //(kg)
    constexpr f64 sol_mass   = 1.98847e30; //(kg)

    /*
    * physical constants
    */
    constexpr f64 c = 299792458;   //(m.s-1)
    constexpr f64 g = 6.67430e-11; //(m^3 kg-1 s-2)


    
} // namespace SI_cst

} // namespace units

template<class flt>
class UnitSystem{ public:

    static_assert(
        std::is_same<flt, f16>::value || std::is_same<flt, f32>::value || std::is_same<flt, f64>::value
    , "UnitSystem : floating point type should be one of (f16,f32,f64)");

    flt to_si_lenght;
    flt to_si_mass;
    flt to_si_time;
    flt to_si_elec_cur;
    flt to_si_temp;
    flt to_si_qte;
    flt to_si_lum_int;

    flt to_si_lenght_2;
    flt to_si_mass_2;
    flt to_si_time_2;
    flt to_si_elec_cur_2;
    flt to_si_temp_2;
    flt to_si_qte_2;
    flt to_si_lum_int_2;

    flt to_si_lenght_3;
    flt to_si_mass_3;
    flt to_si_time_3;
    flt to_si_elec_cur_3;
    flt to_si_temp_3;
    flt to_si_qte_3;
    flt to_si_lum_int_3;


    





    /*
    * lenghts units
    */
    flt m; //(m)
    flt km; //(m)
    flt nm; //(m)
    flt au; //(m)
    flt ly; //(m)
    flt pc; //(m)


    /*
    * time units
    */
    flt s; //(s)
    flt mn; //(s)
    flt hr; //(s)
    flt dy; //(s)
    flt yr; //(s)

    /*
    * mass units
    */
    flt earth_mass; //(kg)
    flt jupiter_mass;   //(kg)
    flt sol_mass;   //(kg)

    /*
    * physical constants
    */
    flt c;  //(m.s-1)
    flt g;  //(m^3 kg-1 s-2)

    #define add_conversion(name,unit) name = units::SI_cst::name/(unit)
    
    inline void update_units(){

        add_conversion(m,to_si_lenght);
        add_conversion(km,to_si_lenght);
        add_conversion(nm,to_si_lenght);
        add_conversion(au,to_si_lenght);
        add_conversion(ly,to_si_lenght);
        add_conversion(pc,to_si_lenght);

        add_conversion(s, to_si_time);
        add_conversion(mn, to_si_time);
        add_conversion(hr, to_si_time);
        add_conversion(dy, to_si_time);
        add_conversion(hr, to_si_time);


        add_conversion(earth_mass,to_si_mass);
        add_conversion(jupiter_mass,to_si_mass);
        add_conversion(sol_mass,to_si_mass);

        add_conversion(c,to_si_lenght/to_si_time);
        add_conversion(g,to_si_lenght_3/(to_si_mass*to_si_time_2));


    }


    inline void set_code_units(
        flt unit_lenght = 1,
        flt unit_mass = 1,
        flt unit_time = 1,
        flt unit_electric_current = 1,
        flt unit_thermo_temperature = 1,
        flt unit_amount_of_substance = 1,
        flt unit_luminous_intensity = 1
        ){

        to_si_lenght     = unit_lenght;
        to_si_mass       = unit_mass;
        to_si_time       = unit_time;
        to_si_elec_cur   = unit_electric_current;
        to_si_temp       = unit_thermo_temperature;
        to_si_qte        = unit_amount_of_substance;
        to_si_lum_int    = unit_luminous_intensity;

        to_si_lenght_2     = to_si_lenght*to_si_lenght;
        to_si_mass_2       = to_si_mass*to_si_mass;
        to_si_time_2       = to_si_time*to_si_time;
        to_si_elec_cur_2   = to_si_elec_cur*to_si_elec_cur;
        to_si_temp_2       = to_si_temp*to_si_temp;
        to_si_qte_2        = to_si_qte*to_si_qte;
        to_si_lum_int_2    = to_si_lum_int*to_si_lum_int;

        to_si_lenght_3     = to_si_lenght*to_si_lenght*to_si_lenght;
        to_si_mass_3       = to_si_mass*to_si_mass*to_si_mass;
        to_si_time_3       = to_si_time*to_si_time*to_si_time;
        to_si_elec_cur_3   = to_si_elec_cur*to_si_elec_cur*to_si_elec_cur;
        to_si_temp_3       = to_si_temp*to_si_temp*to_si_temp;
        to_si_qte_3        = to_si_qte*to_si_qte*to_si_qte;
        to_si_lum_int_3    = to_si_lum_int*to_si_lum_int*to_si_lum_int;

        update_units();
    }


};