
import sys
import struct
import vtk
import vtkmodules



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




def write_dic_to_vtk(dic,filename):

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(dic["x"]))
    for i in range(len(dic["x"])):
        points.SetPoint(i,(dic["x"][i],dic["y"][i],dic["z"][i]))



    ugrid = vtkmodules.vtkCommonDataModel.vtkUnstructuredGrid()

    ugrid.SetPoints(points)


    for k in ["h","omega"]:
        if not k in ["x","y","z"]:
            values = vtk.vtkDoubleArray()
            values.SetName(k)
            values.SetNumberOfValues(len(dic["x"]))
            for i in range(len(dic["x"])):
                values.SetValue(i,dic[k][i])

            ugrid.GetPointData().AddArray(values)


    for desc in [
        ["v","vx","vy","vz"]
        ]:

        vect = vtk.vtkDoubleArray()
        vect.SetName(desc[0])
        vect.SetNumberOfComponents(3)
        vect.SetNumberOfValues(len(dic["x"])*3)

        for i in range(len(dic["x"])):
            vect.SetTuple(i,(dic[desc[1]][i],dic[desc[2]][i],dic[desc[3]][i]))

            ugrid.GetPointData().AddArray(vect)


    for desc in [
        ["a","ax","ay","az"]
        ]:

        vect = vtk.vtkDoubleArray()
        vect.SetName(desc[0])
        vect.SetNumberOfComponents(3)
        vect.SetNumberOfValues(len(dic["x"])*3)

        for i in range(len(dic["x"])):
            vect.SetTuple(i,(dic[desc[1]][i],dic[desc[2]][i],dic[desc[3]][i]))

            ugrid.GetPointData().AddArray(vect)




    fn = filename + '.vtu'
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(fn)
    writer.SetFileTypeToBinary()
    writer.SetInputData(ugrid)
    writer.Write()





import glob
for idx in range(50):

    file_list = glob.glob("./step"+str(idx)+"/patchdata*")

    f = open("./step"+str(idx)+"/timeval.bin","rb")
    tval, = struct.unpack("d",f.read(8))
    f.close()

    
    dic = {}

    for fname in file_list:
        print("converting : {} t = {}".format(fname,tval))
        dic_tmp = get_plot_patchdata(fname)

        for k in dic_tmp.keys():

            if not k in dic.keys():
                dic[k] = []

            dic[k] += dic_tmp[k]

    write_dic_to_vtk(dic,"step"+str(idx))

    lst.append({"timestep" : tval, "file" : "step"+str(idx)+'.vtu'})

    
