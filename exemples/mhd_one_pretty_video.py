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
os.chdir(directory)
root = "dump_"

im_dir = directory + "/prettyimages/"

if not os.path.exists("prettyimages"):
    os.mkdir("prettyimages")
    print(f"Directory '{im_dir}' created.")

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

os.chdir(directory)

if not os.path.exists("images"):
    os.mkdir("images")
    print(f"Directory '{directory}' created.")

for idump in range (ndump):

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

    file = directory + root + f"{idump:04}" + ".sham"
    model.load_from_dump(file)

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap('gist_heat')) # copy the default cmap
    my_cmap.set_bad(color="black")

    vel_arr     =  model.render_cartesian_slice("vxyz","f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    pos_arr     =  model.render_cartesian_slice("xyz","f64_3",  center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    B_on_rho_arr       =  model.render_cartesian_slice("B/rho","f64_3",center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    psi_arr     =  model.render_cartesian_slice("psi/ch","f64", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    uint_arr    =  model.render_cartesian_slice("uint","f64",   center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    rho_arr     =  model.render_cartesian_slice("rho","f64",    center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)

    B_arr = np.zeros_like(B_on_rho_arr)
    B_norm = np.zeros_like(rho_arr)
    B_on_rho_norm = np.zeros_like(rho_arr)
    for i in range(pixel_x):
        for j in range(pixel_y):
            B_arr[i,j,0] = B_on_rho_arr[i,j,0] * rho_arr[i,j]
            B_arr[i,j,1] = B_on_rho_arr[i,j,1] * rho_arr[i,j]
            B_arr[i,j,2] = B_on_rho_arr[i,j,2] * rho_arr[i,j]

            B_norm[i,j] = np.sqrt(B_arr[i,j,0]*B_arr[i,j,0] + B_arr[i,j,1]*B_arr[i,j,1] + B_arr[i,j,2]*B_arr[i,j,2])
            B_on_rho_norm[i,j] = np.sqrt(B_on_rho_arr[i,j,0]*B_on_rho_arr[i,j,0] + B_on_rho_arr[i,j,1]*B_on_rho_arr[i,j,1] + B_on_rho_arr[i,j,2]*B_on_rho_arr[i,j,2])

    dpi=200
    plt.figure(dpi=dpi)
    plt.gca().set_position((0, 0, 1, 1))
    plt.gcf().set_size_inches(pixel_x / dpi, pixel_y / dpi)
    plt.axis('off')

    res = plt.imshow(B_norm, cmap=my_cmap,origin='lower')

    axins = plt.gca().inset_axes([0.73, 0.1, 0.25, 0.025])
    cbar = plt.colorbar(res,cax=axins,orientation="horizontal", extend='both')
    cbar.set_label(r"$\int \rho \, \mathrm{d} z$ [code unit]")

    anchored_text = AnchoredText("t = {:0.3f} [code unit]".format(model.get_time()), loc=2)
    plt.gca().add_artist(anchored_text)

    im_name = "dump_" + f"{idump:04}" + ".png"

    plt.savefig(im_dir + im_name)
    print("image {} saved at {}".format(im_name, im_dir))