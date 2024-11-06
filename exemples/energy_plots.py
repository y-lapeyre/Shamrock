import matplotlib.pyplot as plt
import matplotlib
import shamrock 
import numpy as np
import copy
import subprocess
import os
from matplotlib.offsetbox import AnchoredText

my_cmap = copy.copy(matplotlib.colormaps.get_cmap('gist_heat')) # copy the default cmap
my_cmap.set_bad(color="black")


directory = "/Users/ylapeyre/Documents/Shamwork/fieldloop_horizontal/shamrockdump/"
root = "dump_"
os.chdir(directory)
ndump = 99

pixel_x = 500
pixel_y = 500
radius = 1./2
center = (0,0,0)
cx,cy,cz = center

aspect = pixel_x/pixel_y
pic_range = [-radius*aspect + cx, radius*aspect + cx, -radius + cy, radius + cy]
delta_x = (radius*2*aspect,0.,0.)
delta_y = (0.,radius*2, 0.)
dpi=200

ucin_arr = []
uB_arr = []
Bz_arr = []
time_arr = []

pmass = 4.449388209121246e-05
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 1.,unit_length = 1., unit_mass = 1., )
ucte = shamrock.Constants(codeu)
mu_0 = ucte.mu_0()


bmin = (-0.5, -0.5, -1)
bmax = ( 0.5,  0.5,  1)
xm,ym,zm = bmin
xM,yM,zM = bmax

V = (xM - xm)*(yM - ym)*(zM - zm)
R = 0.3
B0 = 0.001
uB_th = 2 * np.pi * R * R * B0 * B0 / (2 * mu_0 * V)

for idump in range (ndump):

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

    file = directory + root + f"{idump:04}" + ".sham"
    model.load_from_dump(file)

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap('gist_heat')) # copy the default cmap
    my_cmap.set_bad(color="black")

    vel_arr     =  model.render_cartesian_slice("vxyz","f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    B_on_rho_arr =  model.render_cartesian_slice("B/rho","f64_3",center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    rho_arr      =  model.render_cartesian_slice("rho","f64",    center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)

    B_arr = np.zeros_like(B_on_rho_arr)
    uB = 0
    Bz = 0
    ucin = 0

    for i in range(pixel_x):
        for j in range(pixel_y):
            B_arr[i,j,0] = B_on_rho_arr[i,j,0] * rho_arr[i,j]
            B_arr[i,j,1] = B_on_rho_arr[i,j,1] * rho_arr[i,j]
            B_arr[i,j,2] = B_on_rho_arr[i,j,2] * rho_arr[i,j]

            B_norm_sq = B_arr[i,j,0]*B_arr[i,j,0] + B_arr[i,j,1]*B_arr[i,j,1] + B_arr[i,j,2]*B_arr[i,j,2]
            Bz_norm_sq = B_arr[i,j,2]*B_arr[i,j,2]

            uB += B_norm_sq * pmass / rho_arr[i,j]
            Bz += Bz_norm_sq * pmass / rho_arr[i,j]

            ucin += (vel_arr[i,j,0]*vel_arr[i,j,0] + vel_arr[i,j,1]*vel_arr[i,j,1] + vel_arr[i,j,2]*vel_arr[i,j,2])


    uB = uB / (2 * mu_0 * V)
    uB = uB / uB_th
    uB_arr.append(uB)

    uB = uB / (2 * mu_0 * V)
    uB = uB / uB_th
    uB_arr.append(uB)

    
    time_arr.append(model.get_time())

    print("Magnetic energy density for dump {} is {}. The theoretical value is {}".format(idump, uB, uB_th))


fig, axs = plt.subplots(1, 2)
axs[0].plot(time_arr, uB_arr, c='k')
axs[0].set_xlabel("time")
axs[0].set_ylabel("Magnetic energy density")

axs[1].plot(time_arr, uB_arr, c='k')
axs[1].set_xlabel("time")
axs[1].set_ylabel("Bz component")

plt.show()




    