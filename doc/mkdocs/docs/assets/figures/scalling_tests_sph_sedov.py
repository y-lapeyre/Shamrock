
'''
PSMN CascadeLake
s92node01
CPU : Platinum 9242 @ 2.3GHz (48 cores, 2 socket)
RAM : 384 GiB
interconnect 100 GiB/s
'''
X_cascade    = [2,4,8,16]
Rate_Cascade = [981011.0412950808, 2041550.4552234707, 3657283.5229988615, 7577602.18079052]
cnt_Cascade  = [3900960, 7916016, 15835008, 31808472]



'''
grvingt
'''
X_grvingt_1 = [2,4,6,8,10,12,14,16,20,24,28,32,40,52,64,80,100,120]
Rate_grvingt_1 = [
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
]
cnt_grvingt_1  = [
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
]

'''
grvingt 2
'''
X_grvingt_2 = [2,4,6,8,10,12,14,16,20,24,28,32,40,52,64,80,100,120]
Rate_grvingt_2 = [
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
]
cnt_grvingt_2  = [
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
]




import matplotlib.pyplot as plt
import numpy as np
plt.style.use('custom_short_cycler.mplstyle')


plt.figure()
plt.plot(X_cascade  , np.array(Rate_Cascade)  /np.array(X_cascade  ), label = "CascadeLake PSMN")
plt.plot(X_grvingt_1, np.array(Rate_grvingt_1)/np.array(X_grvingt_1), label = "grvingt 1")
plt.plot(X_grvingt_2, np.array(Rate_grvingt_2)/np.array(X_grvingt_2), label = "grvingt 2")
#plt.ylim(0,1e6)
plt.xlim(1,150)
plt.xscale('log')
plt.ylabel(r"$N_{\rm part} / (N_{\rm cpu} t_{\rm step})$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_div.svg")


plt.figure()
plt.plot(X_cascade  , (np.array(Rate_Cascade)  /np.array(X_cascade  ))/(np.array(Rate_Cascade)  /np.array(X_cascade  ))[0], label = "CascadeLake PSMN")
plt.plot(X_grvingt_1, (np.array(Rate_grvingt_1)/np.array(X_grvingt_1))/(np.array(Rate_grvingt_1)/np.array(X_grvingt_1))[0], label = "grvingt 1")
plt.plot(X_grvingt_2, (np.array(Rate_grvingt_2)/np.array(X_grvingt_2))/(np.array(Rate_grvingt_2)/np.array(X_grvingt_2))[0], label = "grvingt 2")
#plt.ylim(0,1e6)
plt.xlim(1,150)
plt.xscale('log')
plt.ylabel(r"$\chi$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_eff.svg")


plt.figure()
plt.plot(X_cascade, np.array(Rate_Cascade)    , label = "CascadeLake PSMN")
plt.plot(X_grvingt_1, np.array(Rate_grvingt_1), label = "grvingt 1")
plt.plot(X_grvingt_2, np.array(Rate_grvingt_2), label = "grvingt 2")
#plt.ylim(0,1e6)
plt.xlim(1,150)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$N_{\rm part} / (t_{\rm step})$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling.svg")