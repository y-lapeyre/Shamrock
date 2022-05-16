
import sys
import struct
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_plot_patchdata(filename):

    f = open(filename,"rb")

    header = (f.read(24))
    header_unpacked = list(struct.unpack("IIIIII", header))

    dic = {}
    
    desc_nam = ["x","y","z"]
    desc_len = [ 4 , 4 , 4 ]
    tot_len = sum(desc_len)

    for a in desc_nam:
        dic[a] = []

    byte = "a"
    for i in range(header_unpacked[0]):
        byte = f.read(tot_len)
        x,y,z = (struct.unpack("fff", byte))

        dic["x"].append(x)
        dic["y"].append(y)
        dic["z"].append(z)



    desc_nam = ["h","omega"]
    desc_len = [ 4 , 4     ]
    tot_len = sum(desc_len)

    for a in desc_nam:
        dic[a] = []

    byte = "a"
    for i in range(header_unpacked[0]):
        byte = f.read(tot_len)
        h,om = (struct.unpack("ff", byte))

        dic["h"].append(h)
        dic["omega"].append(om)





    desc_nam = ["vx","vy","vz","ax","ay","az","ax_old","ay_old","az_old"]
    desc_len = [ 4 , 4  , 4 ,4 , 4  , 4 ,4 , 4  , 4   ]
    tot_len = sum(desc_len)

    for a in desc_nam:
        dic[a] = []

    byte = "a"
    for i in range(header_unpacked[0]):
        byte = f.read(tot_len)
        vx,vy,vz,ax,ay,az,ax_old,ay_old,az_old = (struct.unpack("fffffffff", byte))

        dic["vx"].append(vx)
        dic["vy"].append(vy)
        dic["vz"].append(vz)
        dic["ax"].append(ax)
        dic["ay"].append(ay)
        dic["az"].append(az)
        dic["ax_old"].append(ax_old)
        dic["ay_old"].append(ay_old)
        dic["az_old"].append(az_old)






    f.close()

    dic["x"] = dic["x"][::]
    dic["y"] = dic["y"][::]
    dic["z"] = dic["z"][::]
    dic["h"] = dic["h"][::]

    return dic



def plot_soundwave(idx : int, toff : float):


    file_list = glob.glob("./step"+str(idx)+"/patchdata*")

    f = open("./step"+str(idx)+"/timeval.bin","rb")
    tval, = struct.unpack("d",f.read(8))
    f.close()

    tval -= toff

    dic = {}

    for fname in file_list:
        print("converting : {} t = {}".format(fname,tval))
        dic_tmp = get_plot_patchdata(fname)

        for k in dic_tmp.keys():

            if not k in dic.keys():
                dic[k] = []

            dic[k] += dic_tmp[k]

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, sharex=True) # frameon=False removes frames

    h_arr = np.array(dic["h"])

    m = 1e-5
    hfac = 1.2
    rho = m*(hfac/h_arr)*(hfac/h_arr)*(hfac/h_arr)

    dv0 = 1e-2
    nmode = 2
    z_min = -1
    z_max = 1.0249114036560059
    cs = 1/(1.694)*1.0362095232493131
    z = np.array(dic["z"])

    zs = z_max - z_min

    omega = 2*np.pi*(2/zs)*cs
    print("omega :",omega)
    print("wt/pi :",omega*tval/np.pi)
    anal_vz = dv0*np.cos(nmode*2.*np.pi*(z-z_min)/(z_max-z_min))*np.cos(omega*tval)


    rho_0 = 0.220325
    drho_0 = rho_0*(dv0/cs)

    anal_rho = rho_0 + drho_0*np.sin(nmode*2.*np.pi*(z-z_min)/(z_max-z_min))*np.sin(omega*tval)

    anal_az = -dv0*omega*np.cos(nmode*2.*np.pi*(z-z_min)/(z_max-z_min))*np.sin(omega*tval)

    ax1.scatter(z,dic["vz"],label="SPH")
    ax1.scatter(z,anal_vz,label="analytique")
    ax1.set_xlabel("$z$")
    ax1.set_ylabel("$v_z$")
    ax1.set_title("$t={}$".format(tval))

    ax2.scatter(z,rho,label="SPH")
    ax2.scatter(z,anal_rho,label="analytique")
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("$\\rho$")

    ax3.scatter(z,dic["az"],label="SPH")
    ax3.scatter(z,anal_az,label="analytique")
    ax3.set_xlabel("$z$")
    ax3.set_ylabel("$a_z$")

f = open("./step"+str(4)+"/timeval.bin","rb")
toff, = struct.unpack("d",f.read(8))
f.close()
    
plot_soundwave(5,toff)

plot_soundwave(50,toff)

plot_soundwave(101,toff)

plot_soundwave(150,toff)

plot_soundwave(200,toff)

plot_soundwave(300,toff)

plot_soundwave(400,toff)

plot_soundwave(499,toff)

plt.show()