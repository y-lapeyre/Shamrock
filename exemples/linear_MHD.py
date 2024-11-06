import shamrock
import os
import numpy as np


outputdir = "/Users/ylapeyre/Documents/Shamwork/soundspeedMHD/"
dump_prefix = "linear_"

gamma = 5./3.
rho_g = 1
target_tot_u = 1

C_cour = 0.3
C_force = 0.25

B0 = 1. 
dB0 = 0.00000001
dv0 = 0.00000001
k = np.pi * 4.

dr = 0.01
L = 1.5#0.6 *4
bmin = (-L, -L/2, -L/2)
bmax = ( L,  L/2,  L/2)
xc,yc,zc = 0.,0.,0.
pmass = -1

def B_func(r):
    x,y,z = r
    x = float(x)
    y = float(y)
    z = float(z)
    Bx = B0 + dB0 * np.cos(2 * k * x)
    By = B0 + dB0 * np.cos(2 * k * x)
    Bz = B0 + dB0 * np.cos(2 * k * x)
    

    return (1., 0., 0.)#(Bx, By, Bz)

def vel_func(r):
    
    x,y,z = r
    x = float(x)
    y = float(y)
    z = float(z)
    vx = dv0 * np.cos(2 * k * x)
    vy = dv0 * np.cos(2 * k * x)
    vz = dv0 * np.cos(2 * k * x)
    

    return (0., 0., 0.)

def rho_func(r):
    x,y,z = r
    return np.sin(2 * np.pi *x)

def p_func(r):

    return 1.

def u_func(r):
    u = p_func(r) / (gamma-1)
    return u

def cs_func(r):
    x,y,z = r
    if rho_func(r) == 0.:
        return 0.
    else:
        return np.sqrt(1. / rho_func(r))

def get_dump_name(idump):
    return outputdir + dump_prefix + f"{idump:04}" + ".sham"

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 1.,unit_length = 1., unit_mass = 1., )
ucte = shamrock.Constants(codeu)

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
cfg.set_units(codeu)

#cfg.set_artif_viscosity_None()
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
cfg.set_boundary_periodic()
#cfg.set_boundary_free()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8),1)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

resol = 128
side = 24
#(xs,ys,zs) = model.get_box_dim_fcc_3d(1,resol,24,24)
#dr = 1/xs
#(xs,ys,zs) = model.get_box_dim_fcc_3d(dr,resol,24,24)
(xs,ys,zs) = model.get_box_dim_fcc_3d(1,resol,side, side)
dr = 1/xs
(xs,ys,zs) = model.get_box_dim_fcc_3d(dr,resol,side, side)

model.resize_simulation_box((-xs/2,-ys/2,-zs/2),(xs/2,ys/2,zs/2))
bmin = (-xs/2,-ys/2,-zs/2)
bmax = (xs/2,ys/2,zs/2)
setup = model.get_setup()
gen1 = setup.make_generator_lattice_hcp(dr, (-xs/2,-ys/2,-zs/2),(xs/2,ys/2,zs/2))
#print(comb.get_dot())
setup.apply_setup(gen1)


vol_b = xs*ys*zs
totmass = (rho_g*vol_b) 
print("Total mass :", totmass)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)



#model.set_value_in_a_box("uint","f64", 0 , bmin,bmax)
model.set_field_value_lambda_f64_3("vxyz", vel_func)
model.set_field_value_lambda_f64_3("B/rho", B_func)
#model.set_field_value_lambda_f64("rho", rho_func)
model.set_field_value_lambda_f64("soundspeed", cs_func)

model.set_particle_mass(pmass)

#model.timestep()
model.dump(get_dump_name(0))
model.do_vtk_dump(outputdir + "dump_0000.vtk", True)

t_target = 0.1
dt = 0.000001
t = 0
i = 0



while t<t_target:
    t += dt
    i += 1
    model.evolve_until(t)
    model.do_vtk_dump(outputdir + "dump_{:04}.vtk".format(i), True)
    model.dump(get_dump_name(i))
