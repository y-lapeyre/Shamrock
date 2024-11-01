import matplotlib.pyplot as plt
import matplotlib
import shamrock 
import numpy as np
import copy
import subprocess
import os

import pylab
import sys

np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['text.usetex'] = True #use latex font
my_parameters = {'legend.fontsize':'16','figure.figsize':(8,6),'axes.labelsize':'16','axes.titlesize':'16','xtick.labelsize':'16','ytick.labelsize':'16'}
pylab.rcParams.update(my_parameters) # have bigger text 


directory = "/Users/ylapeyre/Documents/Shamwork/GardnerStone9/shamrockdump/"
folder = "images2D_B"
root = "dump_"


ndump = 66
index = 200

pixel_x = 500
pixel_y = 500
radius = 0.5/2
center = (1.5,0.75,0.75)
cx,cy,cz = center

aspect = pixel_x/pixel_y
pic_range = [-radius*aspect + cx, radius*aspect + cx, -radius + cy, radius + cy]
delta_x = (radius*2*aspect,0.,0.)
delta_y = (0.,0.,radius*2)

os.chdir(directory)

if not os.path.exists(folder):
    os.mkdir(folder)
    print(f"Directory '{directory}' created.")

for idump in range (ndump):

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

    realidump = idump + 0
    file = directory + root + f"{realidump:04}" + ".sham"
    model.load_from_dump(file)

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap('gist_heat')) # copy the default cmap
    my_cmap.set_bad(color="black")

    vel_arr =  model.render_cartesian_column_integ("vxyz", "f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    pos_arr =  model.render_cartesian_column_integ("xyz",  "f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    B_arr   =  model.render_cartesian_column_integ("B/rho",  "f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
    rho_arr =  model.render_cartesian_column_integ("rho","f64",     center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)

    vx_row  = vel_arr[index, :, 0]
    #Bx_row  = B_arr[index, :, 0] * rho_arr[:]
    #By_row  = B_arr[index, :, 1] * rho_arr[:]
    #Bz_row  = B_arr[index, :, 2] * rho_arr[:]
    #Bx_on_rho_row  = B_arr[index, :, 0]
    x_row   = pos_arr[index, :, 0]
    rho_row  = vel_arr[index, :]

    vy_row  = vel_arr[index, :, 1]
    vz_row  = vel_arr[index, :, 2]
  
    Bx_row = []
    By_row = []
    Bz_row = []

    for i in range(pixel_x):
        Bx_on_rho = B_arr[index, i, 0]
        By_on_rho = B_arr[index, i, 1]
        Bz_on_rho = B_arr[index, i, 2]

        Bx = Bx_on_rho * rho_arr[i]
        By = By_on_rho * rho_arr[i]
        Bz = Bz_on_rho * rho_arr[i]

        Bx_row.append(Bx)
        By_row.append(By)
        Bz_row.append(Bz)

    fig, axs = plt.subplots(3, 3, figsize=(12, 6))
    axs[0, 0].plot(x_row, vx_row, c='k')
    axs[0, 0].set_xlabel("$x$ [code unit]")
    axs[0, 0].set_ylabel("$v_x$")
    axs[0, 0].set_ylim(-7*10e-6,7*10e-6)

    axs[0, 1].plot(x_row, vy_row, c='k')
    axs[0, 1].set_xlabel("$x$ [code unit]")
    axs[0, 1].set_ylabel("$v_y$")
    axs[0, 1].set_ylim(-7*10e-12,7*10e-12)

    axs[0, 2].plot(x_row, vz_row, c='k')
    axs[0, 2].set_xlabel("$x$ [code unit]")
    axs[0, 2].set_ylabel("$v_z$ [code unit]")
    axs[0, 2].set_ylim(-7*10e-18,7*10e-18)

    axs[1, 0].plot(x_row, Bx_row, c='k')
    axs[1, 0].set_xlabel("$x$ [code unit]")
    axs[1, 0].set_ylabel("$B_x$")
    axs[1, 0].set_ylim(-7*10e-6,7*10e-6)

    axs[1, 1].plot(x_row, By_row, c='k')
    axs[1, 1].set_xlabel("$x$ [code unit]")
    axs[1, 1].set_ylabel("$B_y$")
    axs[1, 1].set_ylim(-7*10e-6,7*10e-6)

    axs[1, 2].plot(x_row, Bz_row, c='k')
    axs[1, 2].set_xlabel("$x$ [code unit]")
    axs[1, 2].set_ylabel("$B_z$")
    axs[1, 2].set_ylim(-7*10e-6,7*10e-6)

    axs[2, 0].plot(x_row, rho_row, c='k')
    axs[2, 0].set_xlabel("$x$ [code unit]")
    axs[2, 0].set_ylabel("${\rho}$ [code unit]")
    axs[2, 0].set_ylim(-7*10e-6,7*10e-6)

    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("t = {:0.3f} [code unit]".format(model.get_time()))

    im_name = "dump_" + f"{realidump:04}" + ".png"
    im_dir = directory + "/" + folder +  "/"
    plt.savefig(im_dir + im_name)
    print("image {} saved at {}".format(im_name, im_dir))

# Set image input pattern (e.g., img001.png, img002.png...)
input_pattern = "dump_%04d.png"  
output_video = root +".mp4"

# FFmpeg command: Images to video
command = [
    "ffmpeg", 
    "-framerate", "30",  # Frame rate (fps)
    "-i", input_pattern,  # Input file pattern
    "-c:v", "libx264",  # Video codec
    "-pix_fmt", "yuv420p",  # Pixel format for compatibility
    output_video
]

try:
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("FFmpeg is available!")
except FileNotFoundError:
    print("FFmpeg not found! Make sure it's installed and in PATH.")

# Run the FFmpeg command
#subprocess.run(command, check=True)
print("Video created successfully!")