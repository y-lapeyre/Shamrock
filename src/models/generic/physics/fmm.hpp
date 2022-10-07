#include "aliases.hpp"
#include "models/generic/math/tensors.hpp"




template<class T, u32 low_order, u32 high_order>
class GreenFuncGravCartesian{public:
    inline static TensorCollection<T, low_order,high_order> get_der_tensors(sycl::vec<T,3> r);
};


template<class T>
class GreenFuncGravCartesian <T,0,5>{public:
    inline static TensorCollection<T, 0,5> get_der_tensors(sycl::vec<T,3> r){
        T r1 = r.x();
        T r2 = r.y();
        T r3 = r.z();

        T r1pow2 = r.x()*r.x();
        T r2pow2 = r.y()*r.y();
        T r3pow2 = r.z()*r.z();

        T r1pow3 = r.x()*r1pow2;
        T r2pow3 = r.y()*r2pow2;
        T r3pow3 = r.z()*r3pow2;

        T r1pow4 = r.x()*r1pow3;
        T r2pow4 = r.y()*r2pow3;
        T r3pow4 = r.z()*r3pow3;


        T r1pow5 = r.x()*r1pow4;
        T r2pow5 = r.y()*r2pow4;
        T r3pow5 = r.z()*r3pow4;

        T rsq = r1pow2 + r2pow2 + r3pow2;

        T rnorm = sycl::sqrt(rsq);

        T rm2 = 1/(rsq);

        T g0 = 1/rnorm;
        T g1 = -1*rm2*g0;
        T g2 = -3*rm2*g1;
        T g3 = -5*rm2*g2;
        T g4 = -7*rm2*g3;
        T g5 = -9*rm2*g4;

        auto D5 = SymTensor3d_5<T>{
            15*g3*r1 + 10*g4*r1pow3 + g5*r1pow5,
            (3*g3 + 6*g4*r1pow2 + g5*r1pow4)*r2,
            (3*g3 + 6*g4*r1pow2 + g5*r1pow4)*r3,
            3*g3*r1 + g4*r1pow3 + 3*g4*r1*r2pow2 + g5*r1pow3*r2pow2,    
            (3*g4*r1 + g5*r1pow3)*r2*r3,
            3*g3*r1 + g4*r1pow3 + 3*g4*r1*r3pow2 + g5*r1pow3*r3pow2,
            3*g3*r2 + 3*g4*r1pow2*r2 + g4*r2pow3 + g5*r1pow2*r2pow3,    
            (g3 + g4*r1pow2 + g4*r2pow2 + g5*r1pow2*r2pow2)*r3,
            r2*(g3 + g4*r1pow2 + g4*r3pow2 + g5*r1pow2*r3pow2),
            3*g3*r3 + 3*g4*r1pow2*r3 + g4*r3pow3 + g5*r1pow2*r3pow3,    
            r1*(3*g3 + 6*g4*r2pow2 + g5*r2pow4),
            r1*(3*g4*r2 + g5*r2pow3)*r3,
            r1*(g3 + g4*r2pow2 + g4*r3pow2 + g5*r2pow2*r3pow2),
            r1*r2*(3*g4*r3 + g5*r3pow3),    
            r1*(3*g3 + 6*g4*r3pow2 + g5*r3pow4),
            15*g3*r2 + 10*g4*r2pow3 + g5*r2pow5,
            (3*g3 + 6*g4*r2pow2 + g5*r2pow4)*r3,
            3*g3*r2 + g4*r2pow3 + 3*g4*r2*r3pow2 + g5*r2pow3*r3pow2,    
            3*g3*r3 + 3*g4*r2pow2*r3 + g4*r3pow3 + g5*r2pow2*r3pow3,
            r2*(3*g3 + 6*g4*r3pow2 + g5*r3pow4),
            15*g3*r3 + 10*g4*r3pow3 + g5*r3pow5 
        };

        auto D4 = SymTensor3d_4<T>{
            3*g2 + 6*g3*r1pow2 + g4*r1pow4,
            3*g3*r1*r2 + g4*r1pow3*r2,
            3*g3*r1*r3 + g4*r1pow3*r3,
            g2 + g3*r1pow2 + g3*r2pow2 + g4*r1pow2*r2pow2,
            (g3 + g4*r1pow2)*r2*r3,
            g2 + g3*r1pow2 + g3*r3pow2 + g4*r1pow2*r3pow2,
            3*g3*r1*r2 + g4*r1*r2pow3,
            r1*(g3 + g4*r2pow2)*r3,
            r1*r2*(g3 + g4*r3pow2),
            3*g3*r1*r3 + g4*r1*r3pow3,
            3*g2 + 6*g3*r2pow2 + g4*r2pow4,
            3*g3*r2*r3 + g4*r2pow3*r3,
            g2 + g3*r2pow2 + g3*r3pow2 + g4*r2pow2*r3pow2,
            3*g3*r2*r3 + g4*r2*r3pow3,
            3*g2 + 6*g3*r3pow2 + g4*r3pow4
        };


        auto D3 = SymTensor3d_3<T>{
            3*g2*r1 + g3*r1pow3,
            g2*r2 + g3*r1pow2*r2,
            g2*r3 + g3*r1pow2*r3,
            g2*r1 + g3*r1*r2pow2,
            g3*r1*r2*r3,
            g2*r1 + g3*r1*r3pow2,
            3*g2*r2 + g3*r2pow3,
            g2*r3 + g3*r2pow2*r3,
            g2*r2 + g3*r2*r3pow2,
            3*g2*r3 + g3*r3pow3
        };


        auto D2 = SymTensor3d_2<T>{
            g1 + g2*r1pow2,
            g2*r1*r2,
            g2*r1*r3,
            g1 + g2*r2pow2,
            g2*r2*r3,
            g1 + g2*r3pow2
        };
        
        auto D1 = SymTensor3d_1<T>{
            g1*r1,
            g1*r2,
            g1*r3
        };

        auto D0 = g0;

        return TensorCollection<T,0,5>{
            D0,D1,D2,D3,D4,D5
        };
    }
};





template<class T>
class GreenFuncGravCartesian <T,0,4>{public:
    inline static TensorCollection<T, 0,4> get_der_tensors(sycl::vec<T,3> r){
    T r1 = r.x();
    T r2 = r.y();
    T r3 = r.z();

    T r1pow2 = r.x()*r.x();
    T r2pow2 = r.y()*r.y();
    T r3pow2 = r.z()*r.z();

    T r1pow3 = r.x()*r1pow2;
    T r2pow3 = r.y()*r2pow2;
    T r3pow3 = r.z()*r3pow2;

    T r1pow4 = r.x()*r1pow3;
    T r2pow4 = r.y()*r2pow3;
    T r3pow4 = r.z()*r3pow3;

    T rsq = r1pow2 + r2pow2 + r3pow2;

    T rnorm = sycl::sqrt(rsq);

    T rm2 = 1/(rsq);

    T g0 = 1/rnorm;
    T g1 = -1*rm2*g0;
    T g2 = -3*rm2*g1;
    T g3 = -5*rm2*g2;
    T g4 = -7*rm2*g3;


    auto D4 = SymTensor3d_4<T>{
        3*g2 + 6*g3*r1pow2 + g4*r1pow4,
        3*g3*r1*r2 + g4*r1pow3*r2,
        3*g3*r1*r3 + g4*r1pow3*r3,
        g2 + g3*r1pow2 + g3*r2pow2 + g4*r1pow2*r2pow2,
        (g3 + g4*r1pow2)*r2*r3,
        g2 + g3*r1pow2 + g3*r3pow2 + g4*r1pow2*r3pow2,
        3*g3*r1*r2 + g4*r1*r2pow3,
        r1*(g3 + g4*r2pow2)*r3,
        r1*r2*(g3 + g4*r3pow2),
        3*g3*r1*r3 + g4*r1*r3pow3,
        3*g2 + 6*g3*r2pow2 + g4*r2pow4,
        3*g3*r2*r3 + g4*r2pow3*r3,
        g2 + g3*r2pow2 + g3*r3pow2 + g4*r2pow2*r3pow2,
        3*g3*r2*r3 + g4*r2*r3pow3,
        3*g2 + 6*g3*r3pow2 + g4*r3pow4
    };


    auto D3 = SymTensor3d_3<T>{
        3*g2*r1 + g3*r1pow3,
        g2*r2 + g3*r1pow2*r2,
        g2*r3 + g3*r1pow2*r3,
        g2*r1 + g3*r1*r2pow2,
        g3*r1*r2*r3,
        g2*r1 + g3*r1*r3pow2,
        3*g2*r2 + g3*r2pow3,
        g2*r3 + g3*r2pow2*r3,
        g2*r2 + g3*r2*r3pow2,
        3*g2*r3 + g3*r3pow3
    };


    auto D2 = SymTensor3d_2<T>{
        g1 + g2*r1pow2,
        g2*r1*r2,
        g2*r1*r3,
        g1 + g2*r2pow2,
        g2*r2*r3,
        g1 + g2*r3pow2
    };
    
    auto D1 = SymTensor3d_1<T>{
        g1*r1,
        g1*r2,
        g1*r3
    };

    auto D0 = g0;

    return TensorCollection<T,0,4>{
        D0,D1,D2,D3,D4
    };
}
};

template<class T>
class GreenFuncGravCartesian <T,0,3>{public:
    inline static TensorCollection<T, 0,3> get_der_tensors(sycl::vec<T,3> r){
    T r1 = r.x();
    T r2 = r.y();
    T r3 = r.z();

    T r1pow2 = r.x()*r.x();
    T r2pow2 = r.y()*r.y();
    T r3pow2 = r.z()*r.z();

    T r1pow3 = r.x()*r1pow2;
    T r2pow3 = r.y()*r2pow2;
    T r3pow3 = r.z()*r3pow2;

    T rsq = r1pow2 + r2pow2 + r3pow2;

    T rnorm = sycl::sqrt(rsq);

    T rm2 = 1/(rsq);

    T g0 = 1/rnorm;
    T g1 = -1*rm2*g0;
    T g2 = -3*rm2*g1;
    T g3 = -5*rm2*g2;

    auto D3 = SymTensor3d_3<T>{
        3*g2*r1 + g3*r1pow3,
        g2*r2 + g3*r1pow2*r2,
        g2*r3 + g3*r1pow2*r3,
        g2*r1 + g3*r1*r2pow2,
        g3*r1*r2*r3,
        g2*r1 + g3*r1*r3pow2,
        3*g2*r2 + g3*r2pow3,
        g2*r3 + g3*r2pow2*r3,
        g2*r2 + g3*r2*r3pow2,
        3*g2*r3 + g3*r3pow3
    };


    auto D2 = SymTensor3d_2<T>{
        g1 + g2*r1pow2,
        g2*r1*r2,
        g2*r1*r3,
        g1 + g2*r2pow2,
        g2*r2*r3,
        g1 + g2*r3pow2
    };
    
    auto D1 = SymTensor3d_1<T>{
        g1*r1,
        g1*r2,
        g1*r3
    };

    auto D0 = g0;

    return TensorCollection<T,0,3>{
        D0,D1,D2,D3
    };
}
};




template<class T>
class GreenFuncGravCartesian <T,0,2>{public:
    inline static TensorCollection<T, 0,2> get_der_tensors(sycl::vec<T,3> r){
    T r1 = r.x();
    T r2 = r.y();
    T r3 = r.z();

    T r1pow2 = r.x()*r.x();
    T r2pow2 = r.y()*r.y();
    T r3pow2 = r.z()*r.z();

    T r1pow3 = r.x()*r1pow2;
    T r2pow3 = r.y()*r2pow2;
    T r3pow3 = r.z()*r3pow2;

    T rsq = r1pow2 + r2pow2 + r3pow2;

    T rnorm = sycl::sqrt(rsq);

    T rm2 = 1/(rsq);

    T g0 = 1/rnorm;
    T g1 = -1*rm2*g0;
    T g2 = -3*rm2*g1;



    auto D2 = SymTensor3d_2<T>{
        g1 + g2*r1pow2,
        g2*r1*r2,
        g2*r1*r3,
        g1 + g2*r2pow2,
        g2*r2*r3,
        g1 + g2*r3pow2
    };
    
    auto D1 = SymTensor3d_1<T>{
        g1*r1,
        g1*r2,
        g1*r3
    };

    auto D0 = g0;

    return TensorCollection<T,0,2>{
        D0,D1,D2
    };
}
};


template<class T>
class GreenFuncGravCartesian <T,0,1>{public:
    inline static TensorCollection<T, 0,1> get_der_tensors(sycl::vec<T,3> r){
    T r1 = r.x();
    T r2 = r.y();
    T r3 = r.z();

    T r1pow2 = r.x()*r.x();
    T r2pow2 = r.y()*r.y();
    T r3pow2 = r.z()*r.z();

    T r1pow3 = r.x()*r1pow2;
    T r2pow3 = r.y()*r2pow2;
    T r3pow3 = r.z()*r3pow2;

    T rsq = r1pow2 + r2pow2 + r3pow2;

    T rnorm = sycl::sqrt(rsq);

    T rm2 = 1/(rsq);

    T g0 = 1/rnorm;
    T g1 = -1*rm2*g0;
    
    auto D1 = SymTensor3d_1<T>{
        g1*r1,
        g1*r2,
        g1*r3
    };

    auto D0 = g0;

    return TensorCollection<T,0,1>{
        D0,D1
    };
}
};


template<class T>
class GreenFuncGravCartesian <T,0,0>{public:
    inline static TensorCollection<T, 0,0> get_der_tensors(sycl::vec<T,3> r){
    T r1 = r.x();
    T r2 = r.y();
    T r3 = r.z();

    T r1pow2 = r.x()*r.x();
    T r2pow2 = r.y()*r.y();
    T r3pow2 = r.z()*r.z();

    T r1pow3 = r.x()*r1pow2;
    T r2pow3 = r.y()*r2pow2;
    T r3pow3 = r.z()*r3pow2;

    T rsq = r1pow2 + r2pow2 + r3pow2;

    T rnorm = sycl::sqrt(rsq);

    T rm2 = 1/(rsq);

    T g0 = 1/rnorm;
    T g1 = -1*rm2*g0;
    

    auto D0 = g0;

    return TensorCollection<T,0,0>{
        D0
    };
}
};








template<class T, u32 low_order, u32 high_order>
inline TensorCollection<T,low_order,high_order> get_M_mat(TensorCollection<T,low_order,high_order> & D, TensorCollection<T,low_order,high_order> & Q);


template<class T>
inline TensorCollection<T,0,5> get_M_mat(TensorCollection<T,0,5> & D, TensorCollection<T,0,5> & Q){
    T & TD0 = D.t0;
    SymTensor3d_1<T> & TD1 = D.t1;
    SymTensor3d_2<T> & TD2 = D.t2;
    SymTensor3d_3<T> & TD3 = D.t3;
    SymTensor3d_4<T> & TD4 = D.t4;
    SymTensor3d_5<T> & TD5 = D.t5;

    T & TQ0 = Q.t0;
    SymTensor3d_1<T> & TQ1 = Q.t1;
    SymTensor3d_2<T> & TQ2 = Q.t2;
    SymTensor3d_3<T> & TQ3 = Q.t3;
    SymTensor3d_4<T> & TQ4 = Q.t4;
    SymTensor3d_5<T> & TQ5 = Q.t5;


    auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2))*(1/2.) + ((TD3 * TQ3))*(1/6.) + ((TD4 * TQ4))*(1/24.) + (  (TD5 * TQ5))*(1/120.);
    auto M_1 = (-1.)*(TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2))*(1/2.) - ((TD4 * TQ3))*(1/6.) - (  (TD5 * TQ4))*(1/24.);
    auto M_2 = (1./2.)*((TD2 * TQ0) + (TD3 * TQ1) + ((TD4 * TQ2))*(1/2.) + ((TD5 * TQ3))*(1/6.));
    auto M_3 = (1./6.)*((-1.)*(TD3 * TQ0) - (TD4 * TQ1) - ((TD5 * TQ2))*(1/2.));
    auto M_4 = (1./24.)*((TD4 * TQ0) + (TD5 * TQ1));
    auto M_5 = (-1./120)*(TD5 * TQ0);

    return TensorCollection<T, 0, 5>{
        M_0,M_1,M_2,M_3,M_4,M_5
    };

}


template<class T>
inline TensorCollection<T,0,4> get_M_mat(TensorCollection<T,0,4> & D, TensorCollection<T,0,4> & Q){
    T & TD0 = D.t0;
    SymTensor3d_1<T> & TD1 = D.t1;
    SymTensor3d_2<T> & TD2 = D.t2;
    SymTensor3d_3<T> & TD3 = D.t3;
    SymTensor3d_4<T> & TD4 = D.t4;

    T & TQ0 = Q.t0;
    SymTensor3d_1<T> & TQ1 = Q.t1;
    SymTensor3d_2<T> & TQ2 = Q.t2;
    SymTensor3d_3<T> & TQ3 = Q.t3;
    SymTensor3d_4<T> & TQ4 = Q.t4;


    auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2))*(1/2.) + ((TD3 * TQ3))*(1/6.) + ((TD4 * TQ4))*(1/24.);
    auto M_1 = (-1.)*(TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2))*(1/2.) - ((TD4 * TQ3))*(1/6.);
    auto M_2 = (1./2.)*((TD2 * TQ0) + (TD3 * TQ1) + ((TD4 * TQ2))*(1/2.) );
    auto M_3 = (1./6.)*((-1.)*(TD3 * TQ0) - (TD4 * TQ1) );
    auto M_4 = (1./24.)*((TD4 * TQ0));

    return TensorCollection<T, 0, 4>{
        M_0,M_1,M_2,M_3,M_4
    };

}


template<class T>
inline TensorCollection<T,0,3> get_M_mat(TensorCollection<T,0,3> & D, TensorCollection<T,0,3> & Q){
    T & TD0 = D.t0;
    SymTensor3d_1<T> & TD1 = D.t1;
    SymTensor3d_2<T> & TD2 = D.t2;
    SymTensor3d_3<T> & TD3 = D.t3;

    T & TQ0 = Q.t0;
    SymTensor3d_1<T> & TQ1 = Q.t1;
    SymTensor3d_2<T> & TQ2 = Q.t2;
    SymTensor3d_3<T> & TQ3 = Q.t3;


    auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2))*(1/2.) + ((TD3 * TQ3))*(1/6.) ;
    auto M_1 = (-1.)*(TD1 * TQ0) - (TD2 * TQ1) - ((TD3 * TQ2))*(1/2.) ;
    auto M_2 = (1./2.)*((TD2 * TQ0) + (TD3 * TQ1)  );
    auto M_3 = (1./6.)*((-1.)*(TD3 * TQ0)  );

    return TensorCollection<T, 0, 3>{
        M_0,M_1,M_2,M_3
    };

}

template<class T>
inline TensorCollection<T,0,2> get_M_mat(TensorCollection<T,0,2> & D, TensorCollection<T,0,2> & Q){
    T & TD0 = D.t0;
    SymTensor3d_1<T> & TD1 = D.t1;
    SymTensor3d_2<T> & TD2 = D.t2;

    T & TQ0 = Q.t0;
    SymTensor3d_1<T> & TQ1 = Q.t1;
    SymTensor3d_2<T> & TQ2 = Q.t2;


    auto M_0 = (TD0 * TQ0) + (TD1 * TQ1) + ((TD2 * TQ2))*(1/2.)  ;
    auto M_1 = (-1.)*(TD1 * TQ0) - (TD2 * TQ1);
    auto M_2 = (1./2.)*((TD2 * TQ0)  );

    return TensorCollection<T, 0, 2>{
        M_0,M_1,M_2
    };

}

template<class T>
inline TensorCollection<T,0,1> get_M_mat(TensorCollection<T,0,1> & D, TensorCollection<T,0,1> & Q){
    T & TD0 = D.t0;
    SymTensor3d_1<T> & TD1 = D.t1;

    T & TQ0 = Q.t0;
    SymTensor3d_1<T> & TQ1 = Q.t1;


    auto M_0 = (TD0 * TQ0) + (TD1 * TQ1)   ;
    auto M_1 = (-1.)*(TD1 * TQ0) ;

    return TensorCollection<T, 0, 1>{
        M_0,M_1
    };

}

template<class T>
inline TensorCollection<T,0,0> get_M_mat(TensorCollection<T,0,0> & D, TensorCollection<T,0,0> & Q){
    T & TD0 = D.t0;

    T & TQ0 = Q.t0;


    auto M_0 = (TD0 * TQ0) ;

    return TensorCollection<T, 0, 0>{
        M_0
    };

}


template<class T, u32 low_order, u32 high_order>
inline TensorCollection<T,low_order,high_order> offset_multipole(
    const TensorCollection<T,low_order,high_order> & Q_old,
    const sycl::vec<T,3> & offset);

template<class T>
inline TensorCollection<T,0,5> offset_multipole(
    const TensorCollection<T,0,5> & Q,
    const sycl::vec<T,3> & offset){

    TensorCollection<T, 0, 5> d = TensorCollection<T, 0, 5>::from_vec(offset);
    

    auto Qn1 = Q.t1 + Q.t0 * d.t1;



    //mathematica out
    //auto Qn2 = SymTensor3d_2<T>{
    //    Q.t2.v_00 + 2*Q.t1.v_0*d.t1.v_0 + Q.t0*d.t2.v_00,
    //    Q.t2.v_01 + Q.t1.v_1*d.t1.v_0 + Q.t1.v_0*d.t1.v_1 + Q.t0*d.t2.v_01,
    //    Q.t2.v_02 + Q.t1.v_2*d.t1.v_0 + Q.t1.v_0*d.t1.v_2 + Q.t0*d.t2.v_02,
    //    Q.t2.v_11 + 2*Q.t1.v_1*d.t1.v_1 + Q.t0*d.t2.v_11,
    //    Q.t2.v_12 + Q.t1.v_2*d.t1.v_1 + Q.t1.v_1*d.t1.v_2 + Q.t0*d.t2.v_12,
    //    Q.t2.v_22 + 2*Q.t1.v_2*d.t1.v_2 + Q.t0*d.t2.v_22
    //};

    // symbolic Qn2 : d_\mu Qn1_\nu + Q1_\mu d_\nu + Q2
    auto Qn2 = SymTensor3d_2<T>{
        d.t1.v_0 * Qn1.v_0 + Q.t1.v_0 * d.t1.v_0 + Q.t2.v_00,
        d.t1.v_0 * Qn1.v_1 + Q.t1.v_0 * d.t1.v_1 + Q.t2.v_01,
        d.t1.v_0 * Qn1.v_2 + Q.t1.v_0 * d.t1.v_2 + Q.t2.v_02,
        d.t1.v_1 * Qn1.v_1 + Q.t1.v_1 * d.t1.v_1 + Q.t2.v_11,
        d.t1.v_1 * Qn1.v_2 + Q.t1.v_1 * d.t1.v_2 + Q.t2.v_12,
        d.t1.v_2 * Qn1.v_2 + Q.t1.v_2 * d.t1.v_2 + Q.t2.v_22
    };


    
    
    //mathematica out
    //auto Qn3 = SymTensor3d_3<T>{
    //    Q.t3.v_000 + 3*Q.t2.v_00*d.t1.v_0 + 3*Q.t1.v_0*d.t2.v_00 + Q.t0*d.t3.v_000,
    //    Q.t3.v_001 + 2*Q.t2.v_01*d.t1.v_0 + Q.t2.v_00*d.t1.v_1 + Q.t1.v_1*d.t2.v_00 + 2*Q.t1.v_0*d.t2.v_01 + Q.t0*d.t3.v_001,
    //    Q.t3.v_002 + 2*Q.t2.v_02*d.t1.v_0 + Q.t2.v_00*d.t1.v_2 + Q.t1.v_2*d.t2.v_00 + 2*Q.t1.v_0*d.t2.v_02 + Q.t0*d.t3.v_002,
    //    Q.t3.v_011 + Q.t2.v_11*d.t1.v_0 + 2*Q.t2.v_01*d.t1.v_1 + 2*Q.t1.v_1*d.t2.v_01 + Q.t1.v_0*d.t2.v_11 + Q.t0*d.t3.v_011,
    //    Q.t3.v_012 + Q.t2.v_12*d.t1.v_0 + Q.t2.v_02*d.t1.v_1 + Q.t2.v_01*d.t1.v_2 + Q.t1.v_2*d.t2.v_01 + Q.t1.v_1*d.t2.v_02 + Q.t1.v_0*d.t2.v_12 + Q.t0*d.t3.v_012,
    //    Q.t3.v_022 + Q.t2.v_22*d.t1.v_0 + 2*Q.t2.v_02*d.t1.v_2 + 2*Q.t1.v_2*d.t2.v_02 + Q.t1.v_0*d.t2.v_22 + Q.t0*d.t3.v_022,
    //    Q.t3.v_111 + 3*Q.t2.v_11*d.t1.v_1 + 3*Q.t1.v_1*d.t2.v_11 + Q.t0*d.t3.v_111,
    //    Q.t3.v_112 + 2*Q.t2.v_12*d.t1.v_1 + Q.t2.v_11*d.t1.v_2 + Q.t1.v_2*d.t2.v_11 + 2*Q.t1.v_1*d.t2.v_12 + Q.t0*d.t3.v_112,
    //    Q.t3.v_122 + Q.t2.v_22*d.t1.v_1 + 2*Q.t2.v_12*d.t1.v_2 + 2*Q.t1.v_2*d.t2.v_12 + Q.t1.v_1*d.t2.v_22 + Q.t0*d.t3.v_122,
    //    Q.t3.v_222 + 3*Q.t2.v_22*d.t1.v_2 + 3*Q.t1.v_2*d.t2.v_22 + Q.t0*d.t3.v_222
    //};

    // symbolic Qn3 : d_\mu Qn2_\nu\delta + d_\nu * (Qn2_ \mu \delta - d_\mu Qn1_\delta) + Q2_\mu\nu d_\delta + Q3
    auto Qn3 = SymTensor3d_3<T>{
        d.t1.v_0 * Qn2.v_00 + d.t1.v_0 * (Qn2.v_00 - d.t1.v_0 * Qn1.v_0) + Q.t2.v_00 * d.t1.v_0 + Q.t3.v_000,
        d.t1.v_0 * Qn2.v_01 + d.t1.v_0 * (Qn2.v_01 - d.t1.v_0 * Qn1.v_1) + Q.t2.v_00 * d.t1.v_1 + Q.t3.v_001,
        d.t1.v_0 * Qn2.v_02 + d.t1.v_0 * (Qn2.v_02 - d.t1.v_0 * Qn1.v_2) + Q.t2.v_00 * d.t1.v_2 + Q.t3.v_002,
        d.t1.v_0 * Qn2.v_11 + d.t1.v_1 * (Qn2.v_01 - d.t1.v_0 * Qn1.v_1) + Q.t2.v_01 * d.t1.v_1 + Q.t3.v_011,
        d.t1.v_0 * Qn2.v_12 + d.t1.v_1 * (Qn2.v_02 - d.t1.v_0 * Qn1.v_2) + Q.t2.v_01 * d.t1.v_2 + Q.t3.v_012,
        d.t1.v_0 * Qn2.v_22 + d.t1.v_2 * (Qn2.v_02 - d.t1.v_0 * Qn1.v_2) + Q.t2.v_02 * d.t1.v_2 + Q.t3.v_022,
        d.t1.v_1 * Qn2.v_11 + d.t1.v_1 * (Qn2.v_11 - d.t1.v_1 * Qn1.v_1) + Q.t2.v_11 * d.t1.v_1 + Q.t3.v_111,
        d.t1.v_1 * Qn2.v_12 + d.t1.v_1 * (Qn2.v_12 - d.t1.v_1 * Qn1.v_2) + Q.t2.v_11 * d.t1.v_2 + Q.t3.v_112,
        d.t1.v_1 * Qn2.v_22 + d.t1.v_2 * (Qn2.v_12 - d.t1.v_1 * Qn1.v_2) + Q.t2.v_12 * d.t1.v_2 + Q.t3.v_122,
        d.t1.v_2 * Qn2.v_22 + d.t1.v_2 * (Qn2.v_22 - d.t1.v_2 * Qn1.v_2) + Q.t2.v_22 * d.t1.v_2 + Q.t3.v_222
    };

    
    
    auto Qn4 = SymTensor3d_4<T>{
        Q.t4.v_0000 + 4*Q.t3.v_000*d.t1.v_0 + 6*Q.t2.v_00*d.t2.v_00 + 4*Q.t1.v_0*d.t3.v_000 + Q.t0*d.t4.v_0000,
        Q.t4.v_0001 + 3*Q.t3.v_001*d.t1.v_0 + Q.t3.v_000*d.t1.v_1 + 3*Q.t2.v_01*d.t2.v_00 + 3*Q.t2.v_00*d.t2.v_01 + Q.t1.v_1*d.t3.v_000 + 3*Q.t1.v_0*d.t3.v_001 + Q.t0*d.t4.v_0001,
        Q.t4.v_0002 + 3*Q.t3.v_002*d.t1.v_0 + Q.t3.v_000*d.t1.v_2 + 3*Q.t2.v_02*d.t2.v_00 + 3*Q.t2.v_00*d.t2.v_02 + Q.t1.v_2*d.t3.v_000 + 3*Q.t1.v_0*d.t3.v_002 + Q.t0*d.t4.v_0002,
        Q.t4.v_0011 + 2*Q.t3.v_011*d.t1.v_0 + 2*Q.t3.v_001*d.t1.v_1 + Q.t2.v_11*d.t2.v_00 + 4*Q.t2.v_01*d.t2.v_01 + Q.t2.v_00*d.t2.v_11 + 2*Q.t1.v_1*d.t3.v_001 + 2*Q.t1.v_0*d.t3.v_011 + Q.t0*d.t4.v_0011,
        Q.t4.v_0012 + 2*Q.t3.v_012*d.t1.v_0 + Q.t3.v_002*d.t1.v_1 + Q.t3.v_001*d.t1.v_2 + Q.t2.v_12*d.t2.v_00 + 2*Q.t2.v_02*d.t2.v_01 + 2*Q.t2.v_01*d.t2.v_02 + Q.t2.v_00*d.t2.v_12 + Q.t1.v_2*d.t3.v_001 + Q.t1.v_1*d.t3.v_002 + 2*Q.t1.v_0*d.t3.v_012 + Q.t0*d.t4.v_0012,
        Q.t4.v_0022 + 2*Q.t3.v_022*d.t1.v_0 + 2*Q.t3.v_002*d.t1.v_2 + Q.t2.v_22*d.t2.v_00 + 4*Q.t2.v_02*d.t2.v_02 + Q.t2.v_00*d.t2.v_22 + 2*Q.t1.v_2*d.t3.v_002 + 2*Q.t1.v_0*d.t3.v_022 + Q.t0*d.t4.v_0022,
        Q.t4.v_0111 + Q.t3.v_111*d.t1.v_0 + 3*Q.t3.v_011*d.t1.v_1 + 3*Q.t2.v_11*d.t2.v_01 + 3*Q.t2.v_01*d.t2.v_11 + 3*Q.t1.v_1*d.t3.v_011 + Q.t1.v_0*d.t3.v_111 + Q.t0*d.t4.v_0111,
        Q.t4.v_0112 + Q.t3.v_112*d.t1.v_0 + 2*Q.t3.v_012*d.t1.v_1 + Q.t3.v_011*d.t1.v_2 + 2*Q.t2.v_12*d.t2.v_01 + Q.t2.v_11*d.t2.v_02 + Q.t2.v_02*d.t2.v_11 + 2*Q.t2.v_01*d.t2.v_12 + Q.t1.v_2*d.t3.v_011 + 2*Q.t1.v_1*d.t3.v_012 + Q.t1.v_0*d.t3.v_112 + Q.t0*d.t4.v_0112,
        Q.t4.v_0122 + Q.t3.v_122*d.t1.v_0 + Q.t3.v_022*d.t1.v_1 + 2*Q.t3.v_012*d.t1.v_2 + Q.t2.v_22*d.t2.v_01 + 2*Q.t2.v_12*d.t2.v_02 + 2*Q.t2.v_02*d.t2.v_12 + Q.t2.v_01*d.t2.v_22 + 2*Q.t1.v_2*d.t3.v_012 + Q.t1.v_1*d.t3.v_022 + Q.t1.v_0*d.t3.v_122 + Q.t0*d.t4.v_0122,
        Q.t4.v_0222 + Q.t3.v_222*d.t1.v_0 + 3*Q.t3.v_022*d.t1.v_2 + 3*Q.t2.v_22*d.t2.v_02 + 3*Q.t2.v_02*d.t2.v_22 + 3*Q.t1.v_2*d.t3.v_022 + Q.t1.v_0*d.t3.v_222 + Q.t0*d.t4.v_0222,
        Q.t4.v_1111 + 4*Q.t3.v_111*d.t1.v_1 + 6*Q.t2.v_11*d.t2.v_11 + 4*Q.t1.v_1*d.t3.v_111 + Q.t0*d.t4.v_1111,
        Q.t4.v_1112 + 3*Q.t3.v_112*d.t1.v_1 + Q.t3.v_111*d.t1.v_2 + 3*Q.t2.v_12*d.t2.v_11 + 3*Q.t2.v_11*d.t2.v_12 + Q.t1.v_2*d.t3.v_111 + 3*Q.t1.v_1*d.t3.v_112 + Q.t0*d.t4.v_1112,
        Q.t4.v_1122 + 2*Q.t3.v_122*d.t1.v_1 + 2*Q.t3.v_112*d.t1.v_2 + Q.t2.v_22*d.t2.v_11 + 4*Q.t2.v_12*d.t2.v_12 + Q.t2.v_11*d.t2.v_22 + 2*Q.t1.v_2*d.t3.v_112 + 2*Q.t1.v_1*d.t3.v_122 + Q.t0*d.t4.v_1122,
        Q.t4.v_1222 + Q.t3.v_222*d.t1.v_1 + 3*Q.t3.v_122*d.t1.v_2 + 3*Q.t2.v_22*d.t2.v_12 + 3*Q.t2.v_12*d.t2.v_22 + 3*Q.t1.v_2*d.t3.v_122 + Q.t1.v_1*d.t3.v_222 + Q.t0*d.t4.v_1222,
        Q.t4.v_2222 + 4*Q.t3.v_222*d.t1.v_2 + 6*Q.t2.v_22*d.t2.v_22 + 4*Q.t1.v_2*d.t3.v_222 + Q.t0*d.t4.v_2222
    };

    auto Qn5 = SymTensor3d_5<T>{
        Q.t5.v_00000 + 5*Q.t4.v_0000*d.t1.v_0 + 10*Q.t3.v_000*d.t2.v_00 + 10*Q.t2.v_00*d.t3.v_000 + 5*Q.t1.v_0*d.t4.v_0000 + Q.t0*d.t5.v_00000,
        Q.t5.v_00001 + 4*Q.t4.v_0001*d.t1.v_0 + Q.t4.v_0000*d.t1.v_1 + 6*Q.t3.v_001*d.t2.v_00 + 4*Q.t3.v_000*d.t2.v_01 + 4*Q.t2.v_01*d.t3.v_000 + 6*Q.t2.v_00*d.t3.v_001 + Q.t1.v_1*d.t4.v_0000 + 4*Q.t1.v_0*d.t4.v_0001 + Q.t0*d.t5.v_00001,
        Q.t5.v_00002 + 4*Q.t4.v_0002*d.t1.v_0 + Q.t4.v_0000*d.t1.v_2 + 6*Q.t3.v_002*d.t2.v_00 + 4*Q.t3.v_000*d.t2.v_02 + 4*Q.t2.v_02*d.t3.v_000 + 6*Q.t2.v_00*d.t3.v_002 + Q.t1.v_2*d.t4.v_0000 + 4*Q.t1.v_0*d.t4.v_0002 + Q.t0*d.t5.v_00002,
        Q.t5.v_00011 + 3*Q.t4.v_0011*d.t1.v_0 + 2*Q.t4.v_0001*d.t1.v_1 + 3*Q.t3.v_011*d.t2.v_00 + 6*Q.t3.v_001*d.t2.v_01 + Q.t3.v_000*d.t2.v_11 + Q.t2.v_11*d.t3.v_000 + 6*Q.t2.v_01*d.t3.v_001 + 3*Q.t2.v_00*d.t3.v_011 + 2*Q.t1.v_1*d.t4.v_0001 + 3*Q.t1.v_0*d.t4.v_0011 + Q.t0*d.t5.v_00011,
        Q.t5.v_00012 + 3*Q.t4.v_0012*d.t1.v_0 + Q.t4.v_0002*d.t1.v_1 + Q.t4.v_0001*d.t1.v_2 + 3*Q.t3.v_012*d.t2.v_00 + 3*Q.t3.v_002*d.t2.v_01 + 3*Q.t3.v_001*d.t2.v_02 + Q.t3.v_000*d.t2.v_12 + Q.t2.v_12*d.t3.v_000 + 3*Q.t2.v_02*d.t3.v_001 + 3*Q.t2.v_01*d.t3.v_002 + 3*Q.t2.v_00*d.t3.v_012 + Q.t1.v_2*d.t4.v_0001 + Q.t1.v_1*d.t4.v_0002 + 3*Q.t1.v_0*d.t4.v_0012 + Q.t0*d.t5.v_00012,
        Q.t5.v_00022 + 3*Q.t4.v_0022*d.t1.v_0 + 2*Q.t4.v_0002*d.t1.v_2 + 3*Q.t3.v_022*d.t2.v_00 + 6*Q.t3.v_002*d.t2.v_02 + Q.t3.v_000*d.t2.v_22 + Q.t2.v_22*d.t3.v_000 + 6*Q.t2.v_02*d.t3.v_002 + 3*Q.t2.v_00*d.t3.v_022 + 2*Q.t1.v_2*d.t4.v_0002 + 3*Q.t1.v_0*d.t4.v_0022 + Q.t0*d.t5.v_00022,
        Q.t5.v_00111 + 2*Q.t4.v_0111*d.t1.v_0 + 3*Q.t4.v_0011*d.t1.v_1 + Q.t3.v_111*d.t2.v_00 + 6*Q.t3.v_011*d.t2.v_01 + 3*Q.t3.v_001*d.t2.v_11 + 3*Q.t2.v_11*d.t3.v_001 + 6*Q.t2.v_01*d.t3.v_011 + Q.t2.v_00*d.t3.v_111 + 3*Q.t1.v_1*d.t4.v_0011 + 2*Q.t1.v_0*d.t4.v_0111 + Q.t0*d.t5.v_00111,
        Q.t5.v_00112 + 2*Q.t4.v_0112*d.t1.v_0 + 2*Q.t4.v_0012*d.t1.v_1 + Q.t4.v_0011*d.t1.v_2 + Q.t3.v_112*d.t2.v_00 + 4*Q.t3.v_012*d.t2.v_01 + 2*Q.t3.v_011*d.t2.v_02 + Q.t3.v_002*d.t2.v_11 + 2*Q.t3.v_001*d.t2.v_12 + 2*Q.t2.v_12*d.t3.v_001 + Q.t2.v_11*d.t3.v_002 + 2*Q.t2.v_02*d.t3.v_011 + 4*Q.t2.v_01*d.t3.v_012 + Q.t2.v_00*d.t3.v_112 + Q.t1.v_2*d.t4.v_0011 + 2*Q.t1.v_1*d.t4.v_0012 + 2*Q.t1.v_0*d.t4.v_0112 + Q.t0*d.t5.v_00112,
        Q.t5.v_00122 + 2*Q.t4.v_0122*d.t1.v_0 + Q.t4.v_0022*d.t1.v_1 + 2*Q.t4.v_0012*d.t1.v_2 + Q.t3.v_122*d.t2.v_00 + 2*Q.t3.v_022*d.t2.v_01 + 4*Q.t3.v_012*d.t2.v_02 + 2*Q.t3.v_002*d.t2.v_12 + Q.t3.v_001*d.t2.v_22 + Q.t2.v_22*d.t3.v_001 + 2*Q.t2.v_12*d.t3.v_002 + 4*Q.t2.v_02*d.t3.v_012 + 2*Q.t2.v_01*d.t3.v_022 + Q.t2.v_00*d.t3.v_122 + 2*Q.t1.v_2*d.t4.v_0012 + Q.t1.v_1*d.t4.v_0022 + 2*Q.t1.v_0*d.t4.v_0122 + Q.t0*d.t5.v_00122,
        Q.t5.v_00222 + 2*Q.t4.v_0222*d.t1.v_0 + 3*Q.t4.v_0022*d.t1.v_2 + Q.t3.v_222*d.t2.v_00 + 6*Q.t3.v_022*d.t2.v_02 + 3*Q.t3.v_002*d.t2.v_22 + 3*Q.t2.v_22*d.t3.v_002 + 6*Q.t2.v_02*d.t3.v_022 + Q.t2.v_00*d.t3.v_222 + 3*Q.t1.v_2*d.t4.v_0022 + 2*Q.t1.v_0*d.t4.v_0222 + Q.t0*d.t5.v_00222,
        Q.t5.v_01111 + Q.t4.v_1111*d.t1.v_0 + 4*Q.t4.v_0111*d.t1.v_1 + 4*Q.t3.v_111*d.t2.v_01 + 6*Q.t3.v_011*d.t2.v_11 + 6*Q.t2.v_11*d.t3.v_011 + 4*Q.t2.v_01*d.t3.v_111 + 4*Q.t1.v_1*d.t4.v_0111 + Q.t1.v_0*d.t4.v_1111 + Q.t0*d.t5.v_01111,
        Q.t5.v_01112 + Q.t4.v_1112*d.t1.v_0 + 3*Q.t4.v_0112*d.t1.v_1 + Q.t4.v_0111*d.t1.v_2 + 3*Q.t3.v_112*d.t2.v_01 + Q.t3.v_111*d.t2.v_02 + 3*Q.t3.v_012*d.t2.v_11 + 3*Q.t3.v_011*d.t2.v_12 + 3*Q.t2.v_12*d.t3.v_011 + 3*Q.t2.v_11*d.t3.v_012 + Q.t2.v_02*d.t3.v_111 + 3*Q.t2.v_01*d.t3.v_112 + Q.t1.v_2*d.t4.v_0111 + 3*Q.t1.v_1*d.t4.v_0112 + Q.t1.v_0*d.t4.v_1112 + Q.t0*d.t5.v_01112,
        Q.t5.v_01122 + Q.t4.v_1122*d.t1.v_0 + 2*Q.t4.v_0122*d.t1.v_1 + 2*Q.t4.v_0112*d.t1.v_2 + 2*Q.t3.v_122*d.t2.v_01 + 2*Q.t3.v_112*d.t2.v_02 + Q.t3.v_022*d.t2.v_11 + 4*Q.t3.v_012*d.t2.v_12 + Q.t3.v_011*d.t2.v_22 + Q.t2.v_22*d.t3.v_011 + 4*Q.t2.v_12*d.t3.v_012 + Q.t2.v_11*d.t3.v_022 + 2*Q.t2.v_02*d.t3.v_112 + 2*Q.t2.v_01*d.t3.v_122 + 2*Q.t1.v_2*d.t4.v_0112 + 2*Q.t1.v_1*d.t4.v_0122 + Q.t1.v_0*d.t4.v_1122 + Q.t0*d.t5.v_01122,
        Q.t5.v_01222 + Q.t4.v_1222*d.t1.v_0 + Q.t4.v_0222*d.t1.v_1 + 3*Q.t4.v_0122*d.t1.v_2 + Q.t3.v_222*d.t2.v_01 + 3*Q.t3.v_122*d.t2.v_02 + 3*Q.t3.v_022*d.t2.v_12 + 3*Q.t3.v_012*d.t2.v_22 + 3*Q.t2.v_22*d.t3.v_012 + 3*Q.t2.v_12*d.t3.v_022 + 3*Q.t2.v_02*d.t3.v_122 + Q.t2.v_01*d.t3.v_222 + 3*Q.t1.v_2*d.t4.v_0122 + Q.t1.v_1*d.t4.v_0222 + Q.t1.v_0*d.t4.v_1222 + Q.t0*d.t5.v_01222,
        Q.t5.v_02222 + Q.t4.v_2222*d.t1.v_0 + 4*Q.t4.v_0222*d.t1.v_2 + 4*Q.t3.v_222*d.t2.v_02 + 6*Q.t3.v_022*d.t2.v_22 + 6*Q.t2.v_22*d.t3.v_022 + 4*Q.t2.v_02*d.t3.v_222 + 4*Q.t1.v_2*d.t4.v_0222 + Q.t1.v_0*d.t4.v_2222 + Q.t0*d.t5.v_02222,
        Q.t5.v_11111 + 5*Q.t4.v_1111*d.t1.v_1 + 10*Q.t3.v_111*d.t2.v_11 + 10*Q.t2.v_11*d.t3.v_111 + 5*Q.t1.v_1*d.t4.v_1111 + Q.t0*d.t5.v_11111,
        Q.t5.v_11112 + 4*Q.t4.v_1112*d.t1.v_1 + Q.t4.v_1111*d.t1.v_2 + 6*Q.t3.v_112*d.t2.v_11 + 4*Q.t3.v_111*d.t2.v_12 + 4*Q.t2.v_12*d.t3.v_111 + 6*Q.t2.v_11*d.t3.v_112 + Q.t1.v_2*d.t4.v_1111 + 4*Q.t1.v_1*d.t4.v_1112 + Q.t0*d.t5.v_11112,
        Q.t5.v_11122 + 3*Q.t4.v_1122*d.t1.v_1 + 2*Q.t4.v_1112*d.t1.v_2 + 3*Q.t3.v_122*d.t2.v_11 + 6*Q.t3.v_112*d.t2.v_12 + Q.t3.v_111*d.t2.v_22 + Q.t2.v_22*d.t3.v_111 + 6*Q.t2.v_12*d.t3.v_112 + 3*Q.t2.v_11*d.t3.v_122 + 2*Q.t1.v_2*d.t4.v_1112 + 3*Q.t1.v_1*d.t4.v_1122 + Q.t0*d.t5.v_11122,
        Q.t5.v_11222 + 2*Q.t4.v_1222*d.t1.v_1 + 3*Q.t4.v_1122*d.t1.v_2 + Q.t3.v_222*d.t2.v_11 + 6*Q.t3.v_122*d.t2.v_12 + 3*Q.t3.v_112*d.t2.v_22 + 3*Q.t2.v_22*d.t3.v_112 + 6*Q.t2.v_12*d.t3.v_122 + Q.t2.v_11*d.t3.v_222 + 3*Q.t1.v_2*d.t4.v_1122 + 2*Q.t1.v_1*d.t4.v_1222 + Q.t0*d.t5.v_11222,
        Q.t5.v_12222 + Q.t4.v_2222*d.t1.v_1 + 4*Q.t4.v_1222*d.t1.v_2 + 4*Q.t3.v_222*d.t2.v_12 + 6*Q.t3.v_122*d.t2.v_22 + 6*Q.t2.v_22*d.t3.v_122 + 4*Q.t2.v_12*d.t3.v_222 + 4*Q.t1.v_2*d.t4.v_1222 + Q.t1.v_1*d.t4.v_2222 + Q.t0*d.t5.v_12222,
        Q.t5.v_22222 + 5*Q.t4.v_2222*d.t1.v_2 + 10*Q.t3.v_222*d.t2.v_22 + 10*Q.t2.v_22*d.t3.v_222 + 5*Q.t1.v_2*d.t4.v_2222 + Q.t0*d.t5.v_22222
    };

    return {
        Q.t0,
        Qn1,
        Qn2,
        Qn3,
        Qn4,
        Qn5
    };
}

