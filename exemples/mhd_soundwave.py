import shamrock
import matplotlib.pyplot as plt
import numpy as np
import os


directory = "/Users/ylapeyre/Documents/Shamwork/"
outputdir = "native_fast/"

os.chdir(directory)

if not os.path.exists(outputdir):
    os.mkdir(outputdir)
    print(f"Directory '{directory}' created.")
    os.chdir(directory + outputdir)
    os.mkdir("shamrockdump")
    os.mkdir("phantomdump") 


rho_g = 1.

dr = 0.02

bmin = (0, 0 ,  0)#(0, 0, 0)
bmax = (3, 1.5, 1.5)#(L, L/2, L/2)
pmass = -1
gamma = 5./3.

cs= 1.

################################################
################# unit system ##################
################################################
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 1.,unit_length = 1., unit_mass = 1., )
ucte = shamrock.Constants(codeu)

mu_0 = 1.
ctx = shamrock.Context()
ctx.pdata_layout_new()

################################################
#################### config ####################
################################################

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
cfg.set_units(codeu)
cfg.set_artif_viscosity_None()
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
#cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
#cfg.set_eos_isothermal(cs)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e6),1)


################################################
############### size of the box ################
################################################
bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax

model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)
print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)
print("pmass :", pmass)
################################################
############## initial conditions ##############
################################################

A = 1e-4
rho0 = rho_g
wavelength = 1.
k = 2*np.pi/ wavelength#(xM - xm)
va = np.sqrt((1) / (mu_0 * rho0)) #remove mu0

RB_x = 0.
RB_y = 4.#- B0 / va
RB_z = 0.

Rv_x = 4.
Rv_y = -2. #1.
Rv_z = 0.

Rrho = 0.

print("va :", va)
print("mu_0 :", mu_0)

def B_func(r):
    x,y,z = r
    B1 = (1.    + A * RB_x * np.cos(k*x)) 
    B2 = (1.5   + A * RB_y * np.cos(k*x)) 
    B3 = (0.    + A * RB_z * np.cos(k*x)) 
    return (B1, B2, B3)

model.set_field_value_lambda_f64_3("B/rho", B_func)

def vel_func(r):
    x,y,z = r
    v1 = 0.      + A * Rv_x * np.cos(k*x)
    v2 = 0.      + A * Rv_y * np.cos(k*x)
    v3 = 0.      + A * Rv_z * np.cos(k*x)
    return (v1, v2, v3)


model.set_field_value_lambda_f64_3("vxyz", vel_func)

def u_func(r):
    gamma = 5./3.
    P0 = 3./5.
    u = P0 / ((gamma-1) * rho0)
    return u

gamma = 5./3.
P0 = 9
u = P0 / ((gamma-1) * rho0)
model.set_value_in_a_box("uint", "f64", u, bmin,bmax)


print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

t_sum = 0
t_target = 1

i_dump = 0
dt_dump = 0.1 * t_target
next_dt_target = t_sum + dt_dump


while next_dt_target <= t_target *5:

    fname = directory + outputdir + "phantomdump/" + "dump_{:04}.phfile".format(i_dump)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    fnamesh= directory + outputdir + "shamrockdump/" + "dump_" + f"{i_dump:04}" + ".sham"
    model.dump(fnamesh)

    model.evolve_until(next_dt_target)
    

    i_dump += 1

    next_dt_target += dt_dump



