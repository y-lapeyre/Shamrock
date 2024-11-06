import shamrock
import matplotlib.pyplot as plt
import numpy as np
import os


directory = "/Users/ylapeyre/Documents/Shamwork/"
outputdir = "fieldloop_horizontal/"

os.chdir(directory)

if not os.path.exists(outputdir):
    os.mkdir(outputdir)
    print(f"Directory '{directory}' created.")
    os.chdir(directory + outputdir)
    os.mkdir("shamrockdump")
    os.mkdir("phantomdump") 



gamma = 5./3.
rho_g = 1
target_tot_u = 1


dr = 0.01

lambda1 = 2 / np.sqrt(5)
lambda2 = 1.

bmin = (-0.5, -0.5, -1)
bmax = ( 0.5,  0.5,  1)
pmass = -1

R = 0.3
A0 = 1e-3

################################################
################# unit system ##################
################################################
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 1.,unit_length = 1., unit_mass = 1., )
ucte = shamrock.Constants(codeu)

mu_0 = ucte.mu_0()
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
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e6),1)


################################################
############### size of the box ################
################################################
#bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax

model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)
print("pmass :", pmass)

################################################
############## initial conditions ##############
################################################

A = 1e-3
R0 = 0.3
def B_func(r):
    x,y,z = r
    Bx = 0
    By = 0
    Bz = 0

    rnorm = np.sqrt(x*x + y*y)

    if (rnorm <= R):
        Bx = - (A0 / rnorm) * y * mu_0
        By =  (A0 / rnorm) * x * mu_0

    if x==0 and y==0:
        Bx = 0
        By = 0
    

    return (Bx, By, Bz)

model.set_field_value_lambda_f64_3("B/rho", B_func)

def vel_func(r):

    return (1., 0., 0.1 / np.sqrt(5))


model.set_field_value_lambda_f64_3("vxyz", vel_func)

#print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

t_sum = 0
t_target = 1.

i_dump = 0

dt_dump = 0.01
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:

    fname = directory + outputdir + "phantomdump/" + "dump_{:04}.phfile".format(i_dump)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    fnamesh= directory + outputdir + "shamrockdump/" + "dump_" + f"{i_dump:04}" + ".sham"
    model.dump(fnamesh)

    model.evolve_until(next_dt_target)
    

    i_dump += 1

    next_dt_target += dt_dump




