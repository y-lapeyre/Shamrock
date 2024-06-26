import shamrock
import matplotlib.pyplot as plt
import os
import numpy as np
import sarracen

outputdir = "/Users/ylapeyre/Documents/Shamwork/25_06/investigateB/"
ph_file = "/Users/ylapeyre/Documents/Shamwork/25_06/phantom_alfven_fulldump/alfven_00001"
gamma = 5./3.
rho_g = 1
target_tot_u = 1

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

dr = 0.1
L = 0.6 *4
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


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

cfg = model.gen_default_config()
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

###### do a shamrock native setup
#bmin,bmax = model.get_ideal_hcp_box(dr,bmin,bmax)
#xm,ym,zm = bmin
#xM,yM,zM = bmax

###### import a phantom setup

model.resize_simulation_box(bmin,bmax)
#.add_cube_hcp_3d(dr, bmin,bmax)


sdf = sarracen.read_phantom(ph_file)

tuple_of_xyz = (list(sdf['x']), list(sdf['y']), list(sdf['z']))
list_of_xyz = [tuple(item) for item in zip(*tuple_of_xyz)]

tuple_of_v = (list(sdf['vx']), list(sdf['vy']), list(sdf['vz']))
list_of_v = [tuple(item) for item in zip(*tuple_of_v)]

tuple_of_B = (list(sdf['Bx']), list(sdf['By']), list(sdf['Bz']))
list_of_B = [tuple(item) for item in zip(*tuple_of_B)]

model.push_particle(list_of_xyz, list(sdf['h']), list(sdf['u']), list_of_v, list_of_B, list(sdf['psi']))




xc,yc,zc = model.get_closest_part_to((0,0,0))
print("closest part to (0,0,0) is in :",xc,yc,zc)

xm,ym,zm = bmin
xM,yM,zM = bmax
vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

#model.set_value_in_a_box("uint","f64", 0 , bmin,bmax)
#model.set_field_value_lambda_f64_3("vxyz", vel_func)
#model.set_field_value_lambda_f64_3("B/rho", B_func)


rinj = 0.008909042924642563*2
#rinj = 0.008909042924642563*2*2
#rinj = 0.01718181
u_inj = 1
#model.add_kernel_value("uint","f64", u_inj,(0,0,0),rinj) ### @@@@@@@@@ pb: marche que pour uint (pas vxyz, ni B)


tot_u = pmass*model.get_sum("uint","f64")
print("total u :",tot_u)


#print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_length(ctx)



#print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.timestep()
model.do_vtk_dump(outputdir + "dump_0000.vtk", True)
dump = model.make_phantom_dump()
dump.save_dump(outputdir + "init.phdump")

t_target = 0.6#0.1
dt_dump = 0.0000006#0.00001
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


import numpy as np
dic = ctx.collect_data()


if(shamrock.sys.world_rank() == 0):
    

    r = np.sqrt(dic['xyz'][:,0]**2 + dic['xyz'][:,1]**2 +dic['xyz'][:,2]**2)
    vr = np.sqrt(dic['vxyz'][:,0]**2 + dic['vxyz'][:,1]**2 +dic['vxyz'][:,2]**2)


    hpart = dic["hpart"]
    uint = dic["uint"]

    gamma = 5./3.

    rho = pmass*(model.get_hfact()/hpart)**3
    P = (gamma-1) * rho *uint


    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from array_io import *

    fdata = os.path.dirname(os.path.abspath(__file__))+"/sedov_taylor.txt"
    r_theo, rho_theo, p_theo, vr_theo = read_four_arrays(fdata)


    #plt.style.use('custom_style.mplstyle')
    if True:
        
        fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(9,6),dpi=125)

        axs[0,0].scatter(r, vr,c = 'black',s=1,label = "v", rasterized=True)
        axs[0,0].plot(r_theo, vr_theo,c = 'red',label = "v (theory)")
        axs[1,0].scatter(r, uint,c = 'black',s=1,label = "u", rasterized=True)
        axs[0,1].scatter(r, rho,c = 'black',s=1,label = "rho", rasterized=True)
        axs[0,1].plot(r_theo, rho_theo,c = 'red',label = "rho (theory)")
        axs[1,1].scatter(r, P,c = 'black',s=1,label = "P", rasterized=True)
        axs[1,1].plot(r_theo, p_theo,c = 'red',label = "P (theory)")


        axs[0,0].set_ylabel(r"$v$")
        axs[1,0].set_ylabel(r"$u$")
        axs[0,1].set_ylabel(r"$\rho$")
        axs[1,1].set_ylabel(r"$P$")

        axs[0,0].set_xlabel("$r$")
        axs[1,0].set_xlabel("$r$")
        axs[0,1].set_xlabel("$r$")
        axs[1,1].set_xlabel("$r$")

        axs[0,0].set_xlim(0,0.55)
        axs[1,0].set_xlim(0,0.55)
        axs[0,1].set_xlim(0,0.55)
        axs[1,1].set_xlim(0,0.55)
    else:

        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(5,3),dpi=125)

        axs.scatter(r, rho,c = 'black',s=1,label = "rho", rasterized=True)
        axs.plot(r_theo, rho_theo,c = 'red',label = "rho (theory)")

        axs.set_ylabel(r"$\rho$")

        axs.set_xlabel("$r$")

        axs.set_xlim(0,0.55)

    plt.tight_layout()
    plt.legend()
    plt.show()
