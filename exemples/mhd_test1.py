import shamrock
import matplotlib.pyplot as plt
import os
import numpy as np
import sarracen

outputdir = "/Users/ylapeyre/Documents/Shamwork/alfven/"
ph_file = "/Users/ylapeyre/Documents/Phanwork/alfven/alfven_00001"
gamma = 5./3.
rho_g = 1
target_tot_u = 1

C_cour = 0.3
C_force = 0.25

lambda_vel = 1
gamma_vel = 5/3
sina = 0.#2/3
sinb = 0.#np.sqrt(5) / 3
cosa = 1.#2 / np.sqrt(5)
cosb = 1.#1 / np.sqrt(5)
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
pmass = -1
wavelength = 1.
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


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 1.,unit_length = 1., unit_mass = 1., )
ucte = shamrock.Constants(codeu)

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_None()
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.set_units(codeu)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8),1)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.resize_simulation_box(bmin,bmax)
pmass = 9.1362396049896055E-006

sdf = sarracen.read_phantom(ph_file)

tuple_of_xyz = (list(sdf['x']), list(sdf['y']), list(sdf['z']))
list_of_xyz = [tuple(item) for item in zip(*tuple_of_xyz)]

tuple_of_v = (list(sdf['vx']* 10**(-4)), list(sdf['vy']* 10**(-4)), list(sdf['vz']* 10**(-4)))
list_of_v = [tuple(item) for item in zip(*tuple_of_v)]

tuple_of_B = (list(sdf['Bx']*3), list(sdf['By']*3), list(sdf['Bz']*3))
list_of_B = [tuple(item) for item in zip(*tuple_of_B)]

list_of_h = list(sdf['h'])
phmass = 9.1362396049896055E-006 #read from phantom header
hfact = 1.0 #read from phantom header

list_of_rho = [phmass * (hfact / list_of_h[i]) * (hfact / list_of_h[i]) * (hfact / list_of_h[i]) for i in range (len(list_of_h))]
#list_of_arrays_of_B_on_rho = [list_of_B[i] / list_of_rho[i] for i in range (len(list_of_B))]
list_of_B_on_rho = []

for i in range (len(list_of_rho)):
    B_on_rho_x = list_of_B[i][0] / list_of_rho[i]
    B_on_rho_y = list_of_B[i][1] / list_of_rho[i]
    B_on_rho_z = list_of_B[i][2] / list_of_rho[i]
    list_of_B_on_rho.append((B_on_rho_x, B_on_rho_y, B_on_rho_z))

list_of_h = list(sdf['h'])
list_of_u = list(sdf['u'])
list_of_psi = list(sdf['psi'])

# 3 1 1 3 3 1
print(list_of_xyz[0])
print(list_of_h[0])
print(list_of_u[0])
print(list_of_v[0])
print(list_of_B_on_rho[0])
print(list_of_psi[0])


model.push_particle(list_of_xyz, list(sdf['h']), list(sdf['u']), list_of_v, list_of_B_on_rho, list(sdf['psi']))

model.set_particle_mass(pmass)

model.timestep()
model.do_vtk_dump(outputdir + "dump_0000.vtk", True)
dump = model.make_phantom_dump()
dump.save_dump(outputdir + "init.phdump")

t_target = 0.6#0.1
dt_dump = 0.00001
t = 0
i = 0

while t<t_target:
    t += dt_dump
    i += 1
    model.evolve_until(t)
    model.do_vtk_dump(outputdir + "dump_{:04}.vtk".format(i), True)

model.do_vtk_dump(outputdir + "end.vtk", True)
dump = model.make_phantom_dump()
dump.save_dump(outputdir + "end.phdump")
