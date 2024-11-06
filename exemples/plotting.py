import matplotlib.pyplot as plt
import matplotlib
import shamrock 
import numpy as np

file = "/Users/ylapeyre/Documents/Shamwork/GardnerStone7/shamrockdump/dump_0010.sham"
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")
model.load_from_dump(file)

# Reset the figure using the same memory as the last one
import copy
my_cmap = copy.copy(matplotlib.colormaps.get_cmap('gist_heat')) # copy the default cmap
my_cmap.set_bad(color="black")


"""
arr = model.render_cartesian_slice("vxyz","f64_3",center = (0.,0.,0.),delta_x = (2,0,0.),delta_y = (0.,2,0.), nx = 1000, ny = 1000)
res = plt.imshow(arr[:,:,2], cmap=my_cmap,origin='lower')

ax = plt.gca()

plt.xlabel("x")
plt.ylabel("y")
plt.title("t = {:0.3f} [Binary orbit]".format(model.get_time() / (2*np.pi)))

cbar = plt.colorbar(res, extend='both')
cbar.set_label(r"$\rho$ [code unit]")
"""
pixel_x = 1000
pixel_y = 1000
radius = 0.5/2
center = (1.5,0.75,0.75)
cx,cy,cz = center

aspect = pixel_x/pixel_y
pic_range = [-radius*aspect + cx, radius*aspect + cx, -radius + cy, radius + cy]
delta_x = (radius*2*aspect,0.,0.)
delta_y = (0.,radius*2,0.)

vel_arr =  model.render_cartesian_column_integ("vxyz","f64_3",  center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
pos_arr =  model.render_cartesian_column_integ("xyz","f64_3",   center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
B_arr =    model.render_cartesian_column_integ("B/rho","f64_3", center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
psi_arr =  model.render_cartesian_column_integ("psi/ch","f64",  center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
uint_arr = model.render_cartesian_column_integ("uint","f64",    center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
rho_arr =  model.render_cartesian_column_integ("rho","f64",     center = center,delta_x = delta_x,delta_y = delta_y, nx = pixel_x, ny = pixel_y)
#cs_arr =  model.render_cartesian_slice("soundspeed","f64",center = (0.,0.,0.),delta_x = (slice_val,0,0.),delta_y = (0., slice_val,0.), nx = 1000, ny = 1000)

fig, axs = plt.subplots(4, 3, figsize=(10, 6))

im1 = axs[0, 0].imshow(vel_arr[:,:,0], cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im1, ax=axs[0, 0], extend='both')
axs[0, 0].set_title("vx")

im2 = axs[0, 1].imshow(vel_arr[:,:,1], cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im2, ax=axs[0, 1], extend='both')
axs[0, 1].set_title("vy")

im3 = axs[0, 2].imshow(vel_arr[:,:,2], cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im3, ax=axs[0, 2], extend='both')
axs[0, 2].set_title("vz")

im4 = axs[1, 0].imshow(rho_arr, cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im4, ax=axs[1, 0], extend='both')
axs[1, 0].set_title("cs")

im5 = axs[1, 1].imshow(pos_arr[:,:,1], cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im5, ax=axs[1, 1], extend='both')
axs[1, 1].set_title("y")

im6 = axs[1, 2].imshow(pos_arr[:,:,2], cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im6, ax=axs[1, 2], extend='both')
axs[1, 2].set_title("z")

im7 = axs[2, 0].imshow(B_arr[:,:,0], cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im7, ax=axs[2, 0], extend='both')
axs[2, 0].set_title("Bx")

im8 = axs[2, 1].imshow(B_arr[:,:,1], cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im8, ax=axs[2, 1], extend='both')
axs[2, 1].set_title("By")

im9 = axs[2, 2].imshow(B_arr[:,:,2], cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im9, ax=axs[2, 2], extend='both')
axs[2, 2].set_title("Bz")

im10 = axs[3, 0].imshow(psi_arr, cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im9, ax=axs[3, 0], extend='both')
axs[3, 0].set_title("psi")

im11 = axs[3, 1].imshow(uint_arr, cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im11, ax=axs[3, 1], extend='both')
axs[3, 1].set_title("uint")

im12 = axs[3, 2].imshow(rho_arr, cmap=my_cmap, extent=pic_range, origin='lower')
fig.colorbar(im12, ax=axs[3, 2], extend='both')
axs[3, 2].set_title("rho")

plt.tight_layout()



plt.show()