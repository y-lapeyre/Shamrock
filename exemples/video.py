import matplotlib.pyplot as plt
import matplotlib
import shamrock 
import numpy as np
import copy
import subprocess
import os

directory = "/Users/ylapeyre/Documents/Shamwork/shearAlfven2/shamrockdump/"
root = "dump_"


ndump = 68

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
    cs_arr    =  model.render_cartesian_slice("soundspeed","f64",   center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)

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
    
            
    """
    index = 200
    Bx_row = []
    By_row = []
    Bz_row = []
    Bnorm_row  = []

    for i in range(pixel_x):
        Bx_on_rho = B_arr[index, i, 0]
        By_on_rho = B_arr[index, i, 1]
        Bz_on_rho = B_arr[index, i, 2]

        Bx = Bx_on_rho * rho_arr[i]
        By = By_on_rho * rho_arr[i]
        Bz = Bz_on_rho * rho_arr[i]
        Bnorm = np.sqrt(Bx*Bx + By*By + Bz*Bz)

        Bx_row.append(Bx)
        By_row.append(By)
        Bz_row.append(Bz)
        Bnorm_row.append(Bnorm)
    """

    fig, axs = plt.subplots(4, 3, figsize=(10, 6))

    im1 = axs[0, 0].imshow(vel_arr[:,:,0], cmap=my_cmap,origin='lower')
    fig.colorbar(im1, ax=axs[0, 0], extend='both')
    axs[0, 0].set_title("vx")

    im2 = axs[0, 1].imshow(vel_arr[:,:,1], cmap=my_cmap,origin='lower')
    fig.colorbar(im2, ax=axs[0, 1], extend='both')
    axs[0, 1].set_title("vy")

    im3 = axs[0, 2].imshow(vel_arr[:,:,2], cmap=my_cmap,origin='lower')
    fig.colorbar(im3, ax=axs[0, 2], extend='both')
    axs[0, 2].set_title("vz")

    im4 = axs[1, 0].imshow(pos_arr[:,:,0], cmap=my_cmap,origin='lower')
    fig.colorbar(im4, ax=axs[1, 0], extend='both')
    axs[1, 0].set_title("x")

    im5 = axs[1, 1].imshow(B_on_rho_norm, cmap=my_cmap,origin='lower')
    fig.colorbar(im5, ax=axs[1, 1], extend='both')
    axs[1, 1].set_title("B/rho norm")

    im6 = axs[1, 2].imshow(pos_arr[:,:,2], cmap=my_cmap,origin='lower')
    fig.colorbar(im6, ax=axs[1, 2], extend='both')
    axs[1, 2].set_title("z")

    im7 = axs[2, 0].imshow(B_arr[:,:,0], cmap=my_cmap,origin='lower')
    fig.colorbar(im7, ax=axs[2, 0], extend='both')
    axs[2, 0].set_title("Bx")

    im8 = axs[2, 1].imshow(B_arr[:,:,1], cmap=my_cmap,origin='lower')
    fig.colorbar(im8, ax=axs[2, 1], extend='both')
    axs[2, 1].set_title("By")

    im9 = axs[2, 2].imshow(B_arr[:,:,2], cmap=my_cmap,origin='lower')
    fig.colorbar(im9, ax=axs[2, 2], extend='both')
    axs[2, 2].set_title("Bz")

    im10 = axs[3, 0].imshow(psi_arr, cmap=my_cmap,origin='lower')
    fig.colorbar(im9, ax=axs[3, 0], extend='both')
    axs[3, 0].set_title("psi")

    im11 = axs[3, 1].imshow(B_norm, cmap=my_cmap,origin='lower')
    fig.colorbar(im11, ax=axs[3, 1], extend='both')
    axs[3, 1].set_title("Bnorm")

    im12 = axs[3, 2].imshow(rho_arr, cmap=my_cmap,origin='lower')
    fig.colorbar(im12, ax=axs[3, 2], extend='both')
    axs[3, 2].set_title("rho")

    plt.tight_layout()
    im_name = "dump_" + f"{idump:04}" + ".png"
    im_dir = directory + "/images/"
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