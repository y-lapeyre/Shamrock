import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

gamma = 5.0 / 3.0
rho_g = 1
target_tot_u = 1

# grid parameters
base = 64  # resol = base * 2
multx = 1
multy = 1
multz = 1
sz = 1 << 1  # size of the cell
scale_fact = 1 / (sz * base * multx)

center = (base * scale_fact, base * scale_fact, base * scale_fact)
xc, yc, zc = center
rinj = 0.008909042924642563
# rinj = 0.008909042924642563*2*2
# rinj = 0.01718181
u_inj = 1


def uint_map(rmin, rmax):
    x, y, z = rmin
    # print("position in cell unit", x_min, y_min, z_min)
    # print("position", x, y, z)
    x = x - xc  # recenter grid on 0
    y = y - yc
    z = z - zc
    r = np.sqrt(x * x + y * y + z * z)
    if r < rinj:
        return u_inj
    else:
        return 0.0


def rhovel_map(rmin, rmax):
    return (0.0, 0.0, 0.0)


def rho_map(rmin, rmax):
    return 1.0


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(gamma)
cfg.set_riemann_solver_hllc()
cfg.set_slope_lim_vanleer_sym()
cfg.set_face_time_interpolation(True)
cfg.set_Csafe(0.08)
model.set_solver_config(cfg)

model.init_scheduler(int(1e6), 1)

model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", uint_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

model.timestep()
model.dump_vtk("godunov_init2.vtk")
model.dump("godunov_outfile")

t_target = 0.1
model.evolve_until(t_target)
# model.timestep()

model.dump_vtk("godunov_end2.vtk")


dic = ctx.collect_data()
print(dic)

if shamrock.sys.world_rank() == 0:

    x_min = dic["cell_min"][:, 0]
    y_min = dic["cell_min"][:, 1]
    z_min = dic["cell_min"][:, 2]

    x_max = dic["cell_max"][:, 0]
    y_max = dic["cell_max"][:, 1]
    z_max = dic["cell_max"][:, 2]

    x = scale_fact * (x_min + x_max) / 2
    y = scale_fact * (y_min + y_max) / 2
    z = scale_fact * (z_min + z_max) / 2
    x = x - xc  # recenter grid on 0
    y = y - yc
    z = z - zc
    # r = np.sqrt(x * x + y * y + z * z)

    r = []
    for i in range(base * 2):
        x_min = dic["cell_min"][i, 0]
        x_max = dic["cell_max"][i, 0]
        x = scale_fact * (x_min + x_max) / 2
        for j in range(base * 2):
            y_min = dic["cell_min"][j, 1]
            y_max = dic["cell_max"][j, 1]
            y = scale_fact * (y_min + y_max) / 2

            for k in range(base * 2):
                z_min = dic["cell_min"][k, 1]
                z_max = dic["cell_max"][k, 1]
                z = scale_fact * (z_min + z_max) / 2
                r_ijk = np.sqrt(x * x + y * y + z * z)
                r.append(r_ijk)
                # vr_ijk = np.sqrt(dic["rhovel"][i, 0] ** 2 + dic["rhovel"][j, 1] ** 2 + dic["rhovel"][k, 2] ** 2)
                # vr.append(vr_ijk)

    vr = np.sqrt(dic["rhovel"][:, 0] ** 2 + dic["rhovel"][:, 1] ** 2 + dic["rhovel"][:, 2] ** 2)
    uint = dic["rhoetot"]
    rho = dic["rho"]
    P = (gamma - 1) * rho * uint
    print(len(r), len(vr))

    sedov_sol = shamrock.phys.SedovTaylor()

    r_theo = np.linspace(0, 1, 300)
    p_theo = []
    vr_theo = []
    rho_theo = []
    for i in range(len(r_theo)):
        _rho_theo, _vr_theo, _p_theo = sedov_sol.get_value(r_theo[i])
        p_theo.append(_p_theo)
        vr_theo.append(_vr_theo)
        rho_theo.append(_rho_theo)

    r_theo = np.array(r_theo)
    p_theo = np.array(p_theo)
    vr_theo = np.array(vr_theo)
    rho_theo = np.array(rho_theo)

    if True:

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), dpi=125)
        print(len(r), len(vr))
        axs[0, 0].scatter(r, vr, c="black", s=1, label="v", rasterized=True)
        axs[0, 0].plot(r_theo, vr_theo, c="red", label="v (theory)")
        axs[1, 0].scatter(r, uint, c="black", s=1, label="u", rasterized=True)
        axs[0, 1].scatter(r, rho, c="black", s=1, label="rho", rasterized=True)
        axs[0, 1].plot(r_theo, rho_theo, c="red", label="rho (theory)")
        axs[1, 1].scatter(r, P, c="black", s=1, label="P", rasterized=True)
        axs[1, 1].plot(r_theo, p_theo, c="red", label="P (theory)")

        axs[0, 0].set_ylabel(r"$v$")
        axs[1, 0].set_ylabel(r"$u$")
        axs[0, 1].set_ylabel(r"$\rho$")
        axs[1, 1].set_ylabel(r"$P$")

        axs[0, 0].set_xlabel("$r$")
        axs[1, 0].set_xlabel("$r$")
        axs[0, 1].set_xlabel("$r$")
        axs[1, 1].set_xlabel("$r$")

        # axs[0, 0].set_xlim(0, 0.55)
        # axs[1, 0].set_xlim(0, 0.55)
        # axs[0, 1].set_xlim(0, 0.55)
        # axs[1, 1].set_xlim(0, 0.55)
    else:

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=125)

        axs.scatter(r, rho, c="black", s=1, label="rho", rasterized=True)
        axs.plot(r_theo, rho_theo, c="red", label="rho (theory)")

        axs.set_ylabel(r"$\rho$")

        axs.set_xlabel("$r$")

        axs.set_xlim(0, 0.55)

    plt.tight_layout()
    plt.show()
