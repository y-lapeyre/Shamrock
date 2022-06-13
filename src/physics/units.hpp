// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"


namespace units {

    constexpr f64 cm_m = 1e-2;
    constexpr f64 km_m = 1e3;
    constexpr f64 au_m = 149597870700; //(m)
    constexpr f64 ly_m = 9460730472580800; //(m)
    constexpr f64 pc_m = 3.0857e16;//(m)

    constexpr f64 mn_s = 60;//(s)
    constexpr f64 hr_s = 3600;//(s)
    constexpr f64 dy_s = 24*hr_s;//(s)
    constexpr f64 yr_s = 31557600;//(s)

    constexpr f64 g_kg = 1e-3;
    constexpr f64 earth_mass_kg = 5.9722e24; //(kg)
    constexpr f64 jupiter_mass_kg = 1.898e27; //(kg)
    constexpr f64 sol_mass_kg   = 1.98847e30; //(kg)

    constexpr f64 eV_J = 1.602176634e-19; // (J)
    constexpr f64 erg_J = 1e-7; // (J)


    constexpr f64 K_degC_offset = 273.15;

    /*
    * physical constants
    */
    constexpr f64 c_si = 299792458;   //(m.s-1)
    constexpr f64 G_si = 6.67430e-11; //(m^3 kg-1 s-2)
    constexpr f64 kB_si = 1.380649e-23; //(m2 kg s^-2 K^-1 )




    enum UnitEntry{
        s,m,kg,A,K,mol,cd,



    /*
    * alternative base units
    */

    //other times units
    mn,hr,dy,yr,

    //other lenght units
    cm,km,
    au,ly,pc,

    //other mass units
    g,
    earth_mass, jupiter_mass, sol_mass,

    //other e current units

    //other temperature units

    //other quantity units

    //other luminous intensity units




    /*
    * derived units
    */
    Hz 	,///< hertz : frequency (s−1) 	
    N 	,///< (kg⋅m⋅s−2) 	
    Pa 	,///< (kg⋅m−1⋅s−2) 	(N/m2)
    J 	,///< (kg⋅m2⋅s−2) 	(N⋅m = Pa⋅m3)
    W 	,///< (kg⋅m2⋅s−3) 	(J/s)
    C 	,///< (s⋅A) 	
    V 	,///< (kg⋅m2⋅s−3⋅A−1) 	(W/A) = (J/C)
    F 	,///< (kg−1⋅m−2⋅s4⋅A2) 	(C/V) = (C2/J)
    ohm 	,///< (kg⋅m2⋅s−3⋅A−2) 	(V/A) = (J⋅s/C2)
    S 	,///< (kg−1⋅m−2⋅s3⋅A2) 	(ohm−1)
    Wb 	,///< (kg⋅m2⋅s−2⋅A−1) 	(V⋅s)
    T 	,///< (kg⋅s−2⋅A−1) 	(Wb/m2)
    H 	,///< (kg⋅m2⋅s−2⋅A−2) 	(Wb/A)
    degC 	,///< relative to 273.15 K 	(K) 	
    lm 	,///< (cd⋅sr) 	(cd⋅sr)
    lx 	,///< (cd⋅sr⋅m−2) 	(lm/m2)
    Bq 	,///< (s−1) 	
    Gy 	,///< (m2⋅s−2) 	(J/kg)
    Sv 	,///< (m2⋅s−2) 	(J/kg)
    kat 	,///< (mol⋅s−1) 	



    /*
    * alternative derived units
    */
    eV, // (J)
    erg, // (J)
    };

    inline std::string get_symbol(UnitEntry a){

        switch (a) {
            case s            : return "s"            ; break; 
            case m            : return "m"            ; break;
            case kg           : return "kg"           ; break;
            case A            : return "A"            ; break;
            case K            : return "K"            ; break;
            case mol          : return "mol"          ; break;
            case cd           : return "cd"           ; break;
            case mn           : return "mn"           ; break;
            case hr           : return "hr"           ; break;
            case dy           : return "dy"           ; break;
            case yr           : return "yr"           ; break;
            case au           : return "au"           ; break;
            case ly           : return "ly"           ; break;
            case pc           : return "pc"           ; break;
            case earth_mass   : return "earth_mass"   ; break;
            case jupiter_mass : return "jupiter_mass" ; break;
            case sol_mass     : return "sol_mass"     ; break;
            case Hz           : return "Hz"           ; break;
            case N            : return "N"            ; break;
            case Pa           : return "Pa"           ; break;
            case J            : return "J"            ; break;
            case W            : return "W"            ; break;
            case C            : return "C"            ; break;
            case V            : return "V"            ; break;
            case F            : return "F"            ; break;
            case ohm          : return "ohm"          ; break;
            case eV           : return "eV"           ; break;
            case erg          : return "erg"          ; break;
            case cm           : return "cm"           ; break;
            case km           : return "km"           ; break;
            case g            : return "g"            ; break;
            case S            : return "S"            ; break;
            case Wb           : return "Wb"           ; break;
            case T            : return "T"            ; break;
            case H            : return "H"            ; break;
            case degC         : return "degC"         ; break;
            case lm           : return "lm"           ; break;
            case lx           : return "lx"           ; break;
            case Bq           : return "Bq"           ; break;
            case Gy           : return "Gy"           ; break;
            case Sv           : return "Sv"           ; break;
            case kat          : return "kat"          ; break;
                break;
            }

        return "err";
    }


} // namespace units




template<class flt>
class Units{

    static_assert(
        std::is_same<flt, f16>::value || std::is_same<flt, f32>::value || std::is_same<flt, f64>::value
    , "UnitSystem : floating point type should be one of (f16,f32,f64)");
    
    
    
    
    
    public:

    
    


    /*
    * base units
    */
    const flt s,m,kg,A,K,mol,cd;



    /*
    * alternative base units
    */

    //other times units
    const flt mn,hr,dy,yr;

    //other lenght units
    const flt cm,km;
    const flt au,ly,pc;

    //other mass units
    const flt g;
    const flt earth_mass, jupiter_mass, sol_mass;

    //other e current units

    //other temperature units
    //const flt F;  lol no

    //other quantity units

    //other luminous intensity units




    /*
    * derived units
    */
    const flt Hz 	;///< hertz : frequency (s−1) 	
    const flt N 	;///< (kg⋅m⋅s−2) 	
    const flt Pa 	;///< (kg⋅m−1⋅s−2) 	(N/m2)
    const flt J 	;///< (kg⋅m2⋅s−2) 	(N⋅m = Pa⋅m3)
    const flt W 	;///< (kg⋅m2⋅s−3) 	(J/s)
    const flt C 	;///< (s⋅A) 	
    const flt V 	;///< (kg⋅m2⋅s−3⋅A−1) 	(W/A) = (J/C)
    const flt F 	;///< (kg−1⋅m−2⋅s4⋅A2) 	(C/V) = (C2/J)
    const flt ohm 	;///< (kg⋅m2⋅s−3⋅A−2) 	(V/A) = (J⋅s/C2)
    const flt S 	;///< (kg−1⋅m−2⋅s3⋅A2) 	(ohm−1)
    const flt Wb 	;///< (kg⋅m2⋅s−2⋅A−1) 	(V⋅s)
    const flt T 	;///< (kg⋅s−2⋅A−1) 	(Wb/m2)
    const flt H 	;///< (kg⋅m2⋅s−2⋅A−2) 	(Wb/A)
    const flt degC 	;///< relative to 273.15 K 	(K) 	
    const flt lm 	;///< (cd⋅sr) 	(cd⋅sr)
    const flt lx 	;///< (cd⋅sr⋅m−2) 	(lm/m2)
    const flt Bq 	;///< (s−1) 	
    const flt Gy 	;///< (m2⋅s−2) 	(J/kg)
    const flt Sv 	;///< (m2⋅s−2) 	(J/kg)
    const flt kat 	;///< (mol⋅s−1) 	



    /*
    * alternative derived units
    */
    const flt eV; // (J)
    const flt erg; // (J)
    





    /*
    * constants
    */ 
    const flt c;   //(m.s-1)
    const flt G; //(m^3 kg-1 s-2)
    const flt kB; //(m2 kg s^-2 K^-1 )


    Units(
        flt unit_time  ,
        flt unit_lenght  ,
        flt unit_mass ,
        flt unit_current  ,
        flt unit_temperature  ,
        flt unit_qte,
        flt unit_lumint 
        ) :

        s  (1/unit_time  ),
        m  (1/unit_lenght  ),
        kg (1/unit_mass ),
        A  (1/unit_current  ),
        K  (1/unit_temperature  ),
        mol(1/unit_qte),
        cd (1/unit_lumint ),


        //other times units
        mn(s*units::mn_s),
        hr(s*units::hr_s),
        dy(s*units::dy_s),
        yr(s*units::yr_s),

        //other lenght units
        cm(m*units::cm_m),
        km(m*units::km_m),
        au(m*units::au_m),
        ly(m*units::ly_m),
        pc(m*units::pc_m),

        //other mass units
        g(kg*units::g_kg),
        earth_mass(kg*units::earth_mass_kg),
        jupiter_mass(kg*units::jupiter_mass_kg),
        sol_mass(kg*units::sol_mass_kg),

        //other e current units

        //other temperature units

        //other quantity units

        //other luminous intensity units






        /*
        * derived units
        */

        
        Hz 	    (1/s),          // (s−1) 	
        N 	    (kg*m/(s*s)),   // (kg⋅m⋅s−2) 	
        Pa 	    (N / (m*m)),    // (kg⋅m−1⋅s−2) 	(N/m2)
        J 	    (N*m),          // (kg⋅m2⋅s−2) 	(N⋅m = Pa⋅m3)
        W 	    (J/s),          // (kg⋅m2⋅s−3) 	(J/s)
        C 	    (s*A),          // (s⋅A) 	
        V 	    (J/C),          // (kg⋅m2⋅s−3⋅A−1) 	(W/A) = (J/C)
        F 	    (C/V),          // (kg−1⋅m−2⋅s4⋅A2) 	(C/V) = (C2/J)
        ohm 	(V/A),          // (kg⋅m2⋅s−3⋅A−2) 	(V/A) = (J⋅s/C2)
        S 	    (1/ohm),        // (kg−1⋅m−2⋅s3⋅A2) 	(Ω−1)
        Wb 	    (V*s),          // (kg⋅m2⋅s−2⋅A−1) 	(V⋅s)
        T 	    (Wb/(m*m)),     // (kg⋅s−2⋅A−1) 	(Wb/m2)
        H 	    (Wb/A),         // (kg⋅m2⋅s−2⋅A−2) 	(Wb/A)
        degC 	(K - units::K_degC_offset),// relative to 273.15 K 	(K) 	
        lm 	    (cd),           // (cd⋅sr) 	(cd⋅sr)
        lx 	    (cd/(m*m)),     // (cd⋅sr⋅m−2) 	(lm/m2)
        Bq 	    (1/s),          // (s−1) 	
        Gy 	    (m*m/(s*s)),    // (m2⋅s−2) 	(J/kg)
        Sv 	    (m*m/(s*s)),    // (m2⋅s−2) 	(J/kg)
        kat 	(mol/s),        // (mol⋅s−1) 	


        /*
        * alternative derived units
        */
        eV  (J*units::eV_J), // (J)
        erg (J*units::erg_J), // (J)

        /*
        * constants
        */ 
        c (units::c_si * (m/s)),   //(m.s-1)
        G (units::G_si * (m*m*m/(kg*s*s))), //(m^3 kg-1 s-2)
        kB (units::kB_si * (m*m*kg / (s*s*K)))//(m2 kg s^-2 K^-1 )
        
        {}
};

const Units<f64> si_units = Units<f64>(1,1,1,1,1,1,1);


