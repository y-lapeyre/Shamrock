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