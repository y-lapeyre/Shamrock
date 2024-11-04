import shamrock
import matplotlib.pyplot as plt
import os
import numpy as np
import sarracen

outputdir = "/Users/ylapeyre/Documents/Shamwork/alfven_OG/"
dump_prefix = "alfven_artres_"

gamma = 5./3.
rho_g = 1
target_tot_u = 1

C_cour = 0.3
C_force = 0.25

lambda_vel = 1
gamma_vel = 5/3
sina = 2./3.
sinb = 2. / np.sqrt(5)
cosa = np.sqrt(5) / 3.
cosb = 1 / np.sqrt(5)
runit = np.array([cosa*cosb,cosa*sinb,sina])

ampl   = 0.1
przero = 0.1
rhozero = 1.
Bzero  = (1.,0.,0.)
vzero  = 0.
gam1 = gamma_vel-1
uuzero = przero/(gam1*rhozero)

dr = 0.01
L = 1.5#0.6 *4
bmin = (-L, -L/2, -L/2)
bmax = ( L,  L/2,  L/2)
xc,yc,zc = 0.,0.,0.
pmass = -1
wavelength = 0.5
wk = 2.*np.pi/wavelength

def rotated_basis_to_regular(xvec):
    
    reg_xyz = np.array([0., 0., 0.])
    reg_xyz[0] = xvec[0]*cosa*cosb - xvec[1]*sinb - xvec[2]*sina*cosb
    reg_xyz[1] = xvec[0]*cosa*sinb + xvec[1]*cosb - xvec[2]*sina*sinb
    reg_xyz[2] = xvec[0]*sina +      xvec[2]*cosa

    return reg_xyz

def B_func(r):
    #x,y,z = r

    #x1 = cosa * cosb * x + cosa * sinb * y + sina * z
    #x2 = - sinb * x + cosb * y
    #x3 = - sina * cosb * x - sina * sinb * y + cosa * z

    #B1 = 1
    #B2 = 0.1 * np.sin(2 * np.pi * x1 / lambda_vel)
    #B3 = 0.1 * np.cos(2 * np.pi * x1 / lambda_vel)

    #Bx = cosa * cosb * B1 - sinb * B2 - sina * cosb * B3
    #By = cosa * sinb * B1 + cosb * B2 - sina * sinb * B3
    #Bz = sina * B1 + cosa * B3

    x,y,z = r
    x = float(x)
    y = float(y)
    z = float(z)
    rnorm = np.sqrt(x*x + y*y + z*z)
    rvec = np.array([x, y, z])
    #if rnorm ==0:
    #    runit = np.array([0, 0, 0])
    #else:
    #    runit = np.array([x/rnorm, y/rnorm, z/rnorm])

    x0 = np.array((xc, yc, zc))
    xdot0 =np.dot(rvec, x0) 
    x1    = np.dot(rvec, runit)
    sinx1 = np.sin(wk*(x1-xdot0))
    cosx1 = np.cos(wk*(x1-xdot0))

    bvec = Bzero + ampl*np.array([0.,sinx1,cosx1])

    reg_bvec = rotated_basis_to_regular(bvec)
    

    return tuple(reg_bvec)#(Bx, By, Bz)

def vel_func(r):
    x,y,z = r
    x = float(x)
    y = float(y)
    z = float(z)
    rnorm = np.sqrt(x*x + y*y + z*z)
    rvec = np.array([x, y, z])
    #if rnorm ==0:
    #    runit = np.array([0, 0, 0])
    #else:
    #    runit = np.array([x/rnorm, y/rnorm, z/rnorm])

    #x1 = cosa * cosb * x + cosa * sinb * y + sina * z
    #x2 = - sinb * x + cosb * y
    #x3 = - sina * cosb * x - sina * sinb * y + cosa * z

    #v1 = 0
    #v2 = 0.1 * np.sin(2 * np.pi * x1 / lambda_vel)
    #v3 = 0.1 * np.cos(2 * np.pi * x1 / lambda_vel)

    #vx = cosa * cosb * x1 - sinb * x2 - sina * cosb * x3
    #vy = cosa * sinb * x1 + cosb * x2 - sina * sinb * x3
    #vz = sina * x1 + cosa * x3
    x0 = np.array((xc, yc, zc))
    xdot0 =np.dot(rvec, x0) 
    x1    = np.dot(rvec, runit)
    sinx1 = np.sin(wk*(x1-xdot0))
    cosx1 = np.cos(wk*(x1-xdot0))
    vvec = vzero + ampl*np.array((0.,sinx1,cosx1))

    reg_vvec = rotated_basis_to_regular(vvec)

    return tuple(reg_vvec)#(vx, vy, vz)

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
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_None()
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8),1)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

resol = 128
(xs,ys,zs) = model.get_box_dim_fcc_3d(1,resol,24,24)
dr = 1/xs
(xs,ys,zs) = model.get_box_dim_fcc_3d(dr,resol,24,24)

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

model.set_particle_mass(pmass)

model.timestep()
model.dump(get_dump_name(0))
#model.do_vtk_dump(outputdir + "dump_0000.vtk", True)
#dump = model.make_phantom_dump()
#dump.save_dump(outputdir + "init.phdump")

t_target = 0.6#0.1
dt = 0.00001
t = 0
i = 0

model.evolve_once_override_time(0,dt)
model.dump(get_dump_name(1))
model.evolve_once_override_time(dt,dt)
model.dump(get_dump_name(2))

"""
while t<t_target:
    t += dt_dump
    i += 1
    model.evolve_until(t)
    #model.do_vtk_dump(outputdir + "dump_{:04}.vtk".format(i), True)
    model.dump(get_dump_name(i))
"""
