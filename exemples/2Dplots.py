import matplotlib.pyplot as plt
import matplotlib
import shamrock 
import numpy as np
import pylab
import sys

np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['text.usetex'] = True #use latex font
my_parameters = {'legend.fontsize':'16','figure.figsize':(8,6),'axes.labelsize':'16','axes.titlesize':'16','xtick.labelsize':'16','ytick.labelsize':'16'}
pylab.rcParams.update(my_parameters) # have bigger text 


file = "/Users/ylapeyre/Documents/Shamwork/tricco_pushpart5/shamrockdump/dump_0000.sham"
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")
model.load_from_dump(file)

# Reset the figure using the same memory as the last one
import copy
my_cmap = copy.copy(matplotlib.colormaps.get_cmap('gist_heat')) # copy the default cmap
my_cmap.set_bad(color="black")


pixel_x = 1000
pixel_y = 1000
radius = 3/2 /4
center = (0,0,0)
cx,cy,cz = center

aspect = pixel_x/pixel_y
pic_range = [-radius*aspect + cx, radius*aspect + cx, -radius + cy, radius + cy]
delta_x = (0.,radius*2*aspect,0.)
delta_y = (0.,0,radius*2)

vel_arr =  model.render_cartesian_column_integ("vxyz", "f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
pos_arr =  model.render_cartesian_column_integ("xyz",  "f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
B_arr   =  model.render_cartesian_column_integ("B/rho","f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
rho_arr =  model.render_cartesian_column_integ("rho",  "f64",   center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)

B_arr_slice   =  model.render_cartesian_slice("B/rho","f64_3",center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
index = 200
vx_row  = vel_arr[index, :, 0]
x_row   = pos_arr[index, :, 0]

vy_row  = vel_arr[index, :, 1]
y_row   = pos_arr[index, :, 1]

vz_row  = vel_arr[index, :, 2]
z_row   = pos_arr[index, :, 2]

npix = pixel_x
Bx_row = []
By_row = []
Bz_row = []
Bz_row_slice    = []

for i in range(npix):
    Bx_on_rho = B_arr[index, i, 0]
    By_on_rho = B_arr[index, i, 1]
    Bz_on_rho = B_arr[index, i, 2]
    Bz_on_rho_slice = B_arr_slice[index, i, 2]

    Bx = Bx_on_rho * rho_arr[i]
    By = By_on_rho * rho_arr[i]
    Bz = Bz_on_rho * rho_arr[i]
    Bz_slice = Bz_on_rho_slice * rho_arr[i]

    Bx_row.append(Bx)
    By_row.append(By)
    Bz_row.append(Bz)
    Bz_row_slice.append(Bz_slice)



fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs[0, 0].plot(x_row, vx_row, c='k')
axs[0, 0].set_xlabel("$x$")
axs[0, 0].set_ylabel("$v_x$")

axs[0, 1].plot(x_row, vy_row, c='k')
axs[0, 1].set_xlabel("$x$")
axs[0, 1].set_ylabel("$v_y$")

axs[0, 2].plot(x_row, vz_row, c='k')
axs[0, 2].set_xlabel("$x$")
axs[0, 2].set_ylabel("$v_z$")

axs[1, 0].plot(x_row, Bx_row, c='k')
axs[1, 0].set_xlabel("$x$")
axs[1, 0].set_ylabel("$B_x$")

axs[1, 1].plot(x_row, Bz_row_slice, c='k')
axs[1, 1].set_xlabel("$x$")
axs[1, 1].set_ylabel("$B_y$")

axs[1, 2].plot(x_row, Bz_row, c='k')
axs[1, 2].set_xlabel("$x$")
axs[1, 2].set_ylabel("$B_z$")




plt.tight_layout()

plt.show()