#include "accessor.hpp"
#include "aliases.hpp"
#include "buffer.hpp"
#include "builtins.hpp"
#include "models/generic/physics/fmm.hpp"

#include "types.hpp"
#include "unittests/shamrockbench.hpp"
#include "unittests/shamrocktest.hpp"
#include <cstdio>
#include <vector>

#include <random>
#include <iostream>
#include <fstream>

template<class T,u32 order>
class FMM_prec_eval{public:
    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb);
};

template<class T>
class FMM_prec_eval<T,5>{public:
    static constexpr u32 order = 5;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = TensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = TensorCollection<f64,0,order>::from_vec(bj);

                B_n.t0 *= m_j;
                B_n.t1 *= m_j;
                B_n.t2 *= m_j;
                B_n.t3 *= m_j;
                B_n.t4 *= m_j;
                B_n.t5 *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = TensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = TensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;
            f64 phi_2 = M_k.t2*a_k.t2;
            f64 phi_3 = M_k.t3*a_k.t3;
            f64 phi_4 = M_k.t4*a_k.t4;
            f64 phi_5 = M_k.t5*a_k.t5;

            f64 phi_val = phi_0+ phi_1+ phi_2+ phi_3+ phi_4+ phi_5;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }
};



template<class T>
class FMM_prec_eval<T,4>{public:
    static constexpr u32 order = 4;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = TensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = TensorCollection<f64,0,order>::from_vec(bj);

                B_n.t0 *= m_j;
                B_n.t1 *= m_j;
                B_n.t2 *= m_j;
                B_n.t3 *= m_j;
                B_n.t4 *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = TensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = TensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;
            f64 phi_2 = M_k.t2*a_k.t2;
            f64 phi_3 = M_k.t3*a_k.t3;
            f64 phi_4 = M_k.t4*a_k.t4;

            f64 phi_val = phi_0+ phi_1+ phi_2+ phi_3+ phi_4;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }
};




template<class T>
class FMM_prec_eval<T,3>{public:
    static constexpr u32 order = 3;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = TensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = TensorCollection<f64,0,order>::from_vec(bj);

                B_n.t0 *= m_j;
                B_n.t1 *= m_j;
                B_n.t2 *= m_j;
                B_n.t3 *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = TensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = TensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;
            f64 phi_2 = M_k.t2*a_k.t2;
            f64 phi_3 = M_k.t3*a_k.t3;

            f64 phi_val = phi_0+ phi_1+ phi_2+ phi_3;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }
};



template<class T>
class FMM_prec_eval<T,2>{public:
    static constexpr u32 order = 2;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = TensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = TensorCollection<f64,0,order>::from_vec(bj);

                B_n.t0 *= m_j;
                B_n.t1 *= m_j;
                B_n.t2 *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = TensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = TensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;
            f64 phi_2 = M_k.t2*a_k.t2;

            f64 phi_val = phi_0+ phi_1+ phi_2;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }
};






template<class T>
class FMM_prec_eval<T,1>{public:
    static constexpr u32 order = 1;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = TensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = TensorCollection<f64,0,order>::from_vec(bj);

                B_n.t0 *= m_j;
                B_n.t1 *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = TensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = TensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;

            f64 phi_val = phi_0+ phi_1;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }
};



template<class T>
class FMM_prec_eval<T,0>{public:
    static constexpr u32 order = 0;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = TensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = TensorCollection<f64,0,order>::from_vec(bj);

                B_n.t0 *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = TensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = TensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;

            f64 phi_val = phi_0;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }
};





std::string fmm_plot_script = R"==(

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('GTK3Agg')


plt.style.use('custom_style.mplstyle')

res = np.loadtxt("fmm_pot_prec.txt")

res = res[np.argsort(res[:, 0])]


def plot_curve(X,Y,lab):

    ratio = 40

    cnt = len(X)//ratio 

    X_m = X.reshape((cnt,ratio))
    X_m = np.max(X_m,axis=1)

    Y_m = Y.reshape((cnt,ratio))
    Y_m = np.max(Y_m,axis=1)


    plt.plot(X_m,Y_m, label = lab)


plot_curve(res[:,0],np.abs(res[:,6]), "order 0")
plot_curve(res[:,0],np.abs(res[:,5]), "order 1")
plot_curve(res[:,0],np.abs(res[:,4]), "order 2")
plot_curve(res[:,0],np.abs(res[:,3]), "order 3")
plot_curve(res[:,0],np.abs(res[:,2]), "order 4")
plot_curve(res[:,0],np.abs(res[:,1]), "order 5")

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r"$\theta$")

plt.ylabel(r"$\vert \phi_{\rm fmm} - \phi_{\rm th} \vert /\vert \phi_{\rm th}\vert$")

plt.legend()
plt.grid()
plt.savefig("fmm_precision.pdf")

)==";







Test_start("fmm", tmp, 1){


    std::mt19937 eng(0x1111);


    std::uniform_real_distribution<f64> distf64(-1, 1);

    f64 avg_spa = 0.1;
    std::uniform_real_distribution<f64> distf64_red(-avg_spa, avg_spa);


    std::ofstream prec_fmm_pot_out_file("fmm_pot_prec.txt");
    for(u32 i = 0; i < 2e4; i++){


        f64_3 s_a = f64_3{distf64(eng), distf64(eng), distf64(eng)};
        f64_3 s_b = f64_3{distf64(eng), distf64(eng), distf64(eng)};

        f64_3 x_i = s_a + f64_3{distf64_red(eng), distf64_red(eng), distf64_red(eng)};
        f64_3 x_j = s_b + f64_3{distf64_red(eng), distf64_red(eng), distf64_red(eng)};

        f64 angle = (sycl::distance(x_i, s_a) + sycl::distance(x_j, s_b)) / sycl::distance(s_a, s_b);

        prec_fmm_pot_out_file << format(" %e %e %e %e %e %e %e\n"
            ,angle
            ,FMM_prec_eval<f64, 5>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 4>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 3>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 2>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 1>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 0>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b));
    }
    prec_fmm_pot_out_file.close();

    run_py_script(fmm_plot_script);

}


Test_start("fmm", multipole_moment_offset, 1){

    std::mt19937 eng(0x1111);


    std::uniform_real_distribution<f64> distf64(-1, 1);



    //f64_3 s_bp = f64_3{distf64(eng), distf64(eng), distf64(eng)};
    //f64_3 s_b  = f64_3{distf64(eng), distf64(eng), distf64(eng)};

    f64_3 s_bp = f64_3{1, 0, 0};
    f64_3 s_b  = f64_3{0, 0, 0};


    auto B_nB = TensorCollection<f64,0,5>::zeros();
    auto B_nBp = TensorCollection<f64,0,5>::zeros();
    
    for (u32 i = 0; i < 100; i++) {
        
        f64_3 x_1 = f64_3{distf64(eng), distf64(eng), distf64(eng)};

        f64_3 bj = x_1 - s_b;
        f64_3 bpj = x_1 - s_bp;

        auto tB_nB = TensorCollection<f64,0,5>::from_vec(bj);

        

        auto tB_nBp = TensorCollection<f64,0,5>::from_vec(bpj);

        B_nB.t0 += tB_nB.t0;
        B_nB.t1 += tB_nB.t1;
        B_nB.t2 += tB_nB.t2;
        B_nB.t3 += tB_nB.t3;
        B_nB.t4 += tB_nB.t4;
        B_nB.t5 += tB_nB.t5;

        B_nBp.t0 += tB_nBp.t0;
        B_nBp.t1 += tB_nBp.t1;
        B_nBp.t2 += tB_nBp.t2;
        B_nBp.t3 += tB_nBp.t3;
        B_nBp.t4 += tB_nBp.t4;
        B_nBp.t5 += tB_nBp.t5;

    }

    auto d = s_b-s_bp;


    TensorCollection<f64, 0, 5> d_ = TensorCollection<f64, 0, 5>::from_vec(d);

    auto B_nb_offseted = offset_multipole(B_nB,d);

    
    auto diff_0 = B_nBp.t0 - B_nb_offseted.t0;
    auto diff_1 = B_nBp.t1 - B_nb_offseted.t1;
    auto diff_2 = B_nBp.t2 - B_nb_offseted.t2;
    auto diff_3 = B_nBp.t3 - B_nb_offseted.t3;
    auto diff_4 = B_nBp.t4 - B_nb_offseted.t4;
    auto diff_5 = B_nBp.t5 - B_nb_offseted.t5;

    f64 diff = 
        (diff_0*diff_0) + 
        (diff_1*diff_1) + 
        (diff_2*diff_2) + 
        (diff_3*diff_3) + 
        (diff_4*diff_4) + 
        (diff_5*diff_5);

    printf("order 0 %f\n",B_nB.t0);




    printf("diff %e\n",diff);

    printf("dvec   %f %f %f\n",d.x(),d.y(),d.z());

    printf("np     %f %f %f\n",B_nBp.t1.v_0,B_nBp.t1.v_1,B_nBp.t1.v_2);
    printf("B      %f %f %f\n",B_nB.t1.v_0,B_nB.t1.v_1,B_nB.t1.v_2);
    printf("offset %f %f %f\n",B_nb_offseted.t1.v_0,B_nb_offseted.t1.v_1,B_nb_offseted.t1.v_2);
    
    printf("------ order 2 ---------\n");
    printf("B      %f %f %f %f %f %f\n",B_nB.t2.v_00,B_nB.t2.v_01,B_nB.t2.v_02,B_nB.t2.v_11,B_nB.t2.v_12,B_nB.t2.v_22);
    printf("np     %f %f %f %f %f %f\n",B_nBp.t2.v_00,B_nBp.t2.v_01,B_nBp.t2.v_02,B_nBp.t2.v_11,B_nBp.t2.v_12,B_nBp.t2.v_22);
    printf("offset %f %f %f %f %f %f\n",B_nb_offseted.t2.v_00,B_nb_offseted.t2.v_01,B_nb_offseted.t2.v_02,B_nb_offseted.t2.v_11,B_nb_offseted.t2.v_12,B_nb_offseted.t2.v_22);
    printf("d      %f %f %f %f %f %f\n",d_.t2.v_00,d_.t2.v_01,d_.t2.v_02,d_.t2.v_11,d_.t2.v_12,d_.t2.v_22);
    

    auto DG = GreenFuncGravCartesian<f64, 0, 5>::get_der_tensors(s_b - f64_3{5,1,2});

    auto err = (DG.t2 * (B_nb_offseted.t1 % d_.t1)) - ( 2*(B_nb_offseted.t1 * DG.t2 * d_.t1));

    printf("%f\n ", err*err);

    printf("%e %e %e %e %e %e\n",(diff_0*diff_0) , 
        (diff_1*diff_1) , 
        (diff_2*diff_2) , 
        (diff_3*diff_3) , 
        (diff_4*diff_4) , 
        (diff_5*diff_5));
}

Bench_start("fmm shit", "fmm performance", fmm_perf, 1){
    Register_score(1)



    
    
    
    std::vector<f64_3> pos_table = {{1,1,1}};

    sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());


    constexpr u32 order = 5;



    constexpr u32 number_elem_multip = TensorCollection<f64,0,order>::num_component;
    sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


    //compute multipoles
    auto l = [&]{
        sycl::host_accessor<f64_3> pos {buf_pos};
        sycl::host_accessor<f64> multipoles {buf_multipoles};

        f64_3 sa = {0,0,0};
        f64_3 sb = {0,0,0};

        for (u32 j = 0; j < pos_table.size(); j ++) {

            f64_3 xj = pos[j];

            f64_3 bj = xj - sb;

            auto B_n = TensorCollection<f64,0,order>::from_vec(bj);


            
        }

    };

    TimeitFor(100,l())

}
