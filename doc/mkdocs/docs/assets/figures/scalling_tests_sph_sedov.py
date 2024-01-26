

dic = {}


'''
PSMN CascadeLake
s92node01
CPU : Platinum 9242 @ 2.3GHz (48 cores, 2 socket)
RAM : 384 GiB
interconnect 100 GiB/s
'''
#dic["CascadeLake1"] = {
#    "label" : "CascadeLake",
#    "X" : [2,4,8,16,128],
#    "rate" : [981011.0412950808, 2041550.4552234707, 3657283.5229988615, 7577602.18079052, 42975622.415431604],
#    "cnt" : [3900960, 7916016, 15835008, 31808472, 254467776],
#    "lb_pred" : [100,100,100,100, 100]
#}

dic["CascadeLake1"] = {
    "label" : "CascadeLake",
    "X" : [2, 4, 8, 16, 32,64,128,152],
    "rate" : [1705829.042131229, 3328780.0407591304, 6709830.7891846, 10790180.093741834, 20591599.941228684,40850996.51509906, 63314184.38101712,76556249.52454272],
    "cnt" : [11838528, 23793664, 47434464, 95496480, 191092864,381677472, 767134656,910455552],
    "lb_pred" : [100, 100, 100, 100,100,100,100, 100]
}



'''
grvingt
'''
dic["grvingt 1"] = {
    "label" : "grvingt ",
    "X" : [2,4,6,8,10,12,14,16,20,24,28,32,40,52,64,80,100,120],
    "rate" : [
        591498.6491027079,
        1181681.1136507655,
        1553104.688674408,
        2116310.7803585893,
        2593814.229834617,
        3168616.5402065823,
        2383392.539410924,
        4684412.440914724,
        4846178.7416838575,
        6477531.774039453,
        6464480.133897541,
        5853629.434544097,
        9719550.417095121,
        10949372.839548606,
        10947479.134140132,
        13665466.536836417,
        15941812.629143752,
        19827022.50752665
    ],
    "cnt" : [
        3900960, 
        7916016,
        11838528,
        15835008,
        19749120,
        23793664,
        27799200,
        31808472,
        39580800,
        47434464,
        55444480,
        63638560,
        79356680,
        103155712,
        127635200,
        159221952,
        199056000,
        238775328
    ],
    "lb_pred" : [
        99.99, 
        97.92,
        90.08,
        100,
        84.96,
        82.57,
        78.57,
        98.59,
        72.71,
        66.19,
        65.20,
        88.55,
        90.54,
        88.59,
        97.64,
        83.19,
        81.68,
        78.51
    ]
}

'''
grvingt 2
'''

dic["grvingt 2"] = {
    "label" : "grvingt 2",
    "X" : [2,4,6,8,10,12,14,16,20,24,28,32,40,52,64,80,100,120],
    "rate" : [
        628358.5779473973,
        1124054.859004319,
        1736449.7664354416,
        2380810.3909499245,
        1917293.5540002272,
        3382532.054667773,
        3894101.199054512,
        4504015.621482006,
        5211853.5672429325,
        4301093.504322428,
        7283223.907001199,
        8527158.81533037,
        11048942.191861987,
        10292658.651045002,
        17199615.303962123,
        21550971.728999294,
        26018819.0099095,
        31249213.904896423
    ],
    "cnt" : [
        7916016, 
        15835008,
        23793664,
        31808472,
        39580800,
        47434464,
        55444480,
        63638560,
        79356680,
        95496480,
        111213232,
        127635200,
        159221952,
        206595968,
        254467776,
        318404736,
        399787392,
        477884736
    ],
    "lb_pred" : [
        98.88, 
        100,
        90.73,
        100,
        85.07,
        83.24,
        79.11,
        93.87,
        95.08,
        94.33,
        93.31,
        98.35,
        91.49,
        88.90,
        78.70,
        84.19,
        81.81,
        78.99
    ]
}


'''
grvingt 3
'''
dic["grvingt 3"] = {
    "label" : "grvingt 3",
    "X" : [2,4,6,8,10,12,14,16,20,24,28,32,40,52,64,80,100,120],

    "rate" : [

        592141.0651991414,
        1239845.601482153,
        1858803.6876634469,
        2379322.316033665,
        2720184.7882375405,
        3328598.2052814085,
        3837423.3467434007,
        4464796.424425585,
        5782054.636144191,
        7018958.537090696,
        8070815.635627935,
        9457773.055821262,
        10776156.334684804,
        15829854.006017553,
        17850911.895728435,
        20357623.0642507,
        25717710.869283035,
        31210662.41445782

    ],
    "cnt" : [

        15835008,
        31808472,
        47434464,
        63638560,
        79356680,
        95496480,
        111213232,
        127635200,
        159221952,
        191092864,
        223067520,
        254467776,
        318404736,
        415100928,
        510350208,
        637955440,
        797460048,
        957954192

    ],
    "lb_pred" : [
        100, 
        100,
        91,
        97.45,
        97.69,
        96.73,
        96.11,
        99.11,
        95.68,
        94.09,
        93.12,
        88.72,
        91.28,
        89.28,
        79.75,
        96.94,
        95.79,
        94.75
    ]

}


import matplotlib.pyplot as plt
import numpy as np
plt.style.use('custom_short_cycler.mplstyle')


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ), label = dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.ylabel(r"$N_{\rm part} / (N_{\rm cpu} t_{\rm step})$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_div_cpu.svg")


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , (np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ))/(np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ))[0], label =  dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.ylabel(r"$\chi$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_eff_cpu.svg")


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , np.array(dic[k]["rate"]), label = dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$N_{\rm part} / (t_{\rm step})$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_cpu.svg")





dic = {}
dic["Adastra mi250X"] = {
    "label" : "Adastra mi250X",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4],
    "rate" : [
        28144452.5247059, 
        65301034.81380231, 
        126258518.4939734, 
        220797309.04850662, 
        451664700.5182303, 
        850424568.3261222,
        1548188962.967435],
    "cnt" : [
        403064480, 
        802649952, 
        1606415328, 
        3211389720, 
        6421199616,
        12828966752, 
        25658246736],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ), label = dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.ylabel(r"$N_{\rm part} / (N_{\rm GPU} t_{\rm step})$")
plt.xlabel(r"$N_{\rm GPU}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_div_GPU.svg")


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , (np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ))/(np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ))[0], label =  dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.ylabel(r"$\chi$")
plt.xlabel(r"$N_{\rm GPU}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_eff_GPU.svg")


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , np.array(dic[k]["rate"]), label = dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$N_{\rm part} / (t_{\rm step})$")
plt.xlabel(r"$N_{\rm GPU}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_GPU.svg")