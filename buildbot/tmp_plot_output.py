import mayavi
import mayavi.mlab
import sys
import struct
import matplotlib.pyplot as plt
# mayavi.mlab.options.backend = 'envisage'

#mayavi.mlab.figure(figure=None, bgcolor=None, fgcolor=None, engine=None, size=(1920, 1080))

def plot_patchdata(filename):

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

    f.close()

    #dic["x"] = dic["x"][::10]
    #dic["y"] = dic["y"][::10]
    #dic["z"] = dic["z"][::10]
    #dic["h"] = dic["h"][::10]

    #print(dic["h"][:1000])

    #mayavi.mlab.points3d(dic["x"],dic["y"],dic["z"],dic["h"], scale_factor=0.5)

    dic_filtered = {
        "x" : [],
        "y" : [],
        "z" : [],
        "h" : [],
    }

    for i in range(len(dic["x"])):

        cd = True
        #cd = cd and dic["y"][i] < 0.1 and dic["y"][i] > -0.1
        cd = cd and dic["z"][i] < 0.3e-2 and dic["z"][i] > -0.3e-2


        if cd :

            dic_filtered["x"].append(dic["x"][i])
            dic_filtered["y"].append(dic["y"][i])
            dic_filtered["z"].append(dic["z"][i])
            dic_filtered["h"].append(dic["h"][i])


    plt.scatter(dic_filtered["x"], dic_filtered["y"], c=dic_filtered["h"], cmap='nipy_spectral',vmin=0,vmax=0.025)



def plot_content(filename):

    print("plotting : {}".format(filename))

    if "patchdata" in filename:
        plot_patchdata(filename)

for a in sys.argv[2::]:
    plot_content(a)

plt.colorbar()

plt.show()
#mayavi.mlab.show()
#print("saving ...")
#mayavi.mlab.savefig(sys.argv[1])