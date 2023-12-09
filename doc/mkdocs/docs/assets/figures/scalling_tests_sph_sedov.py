
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

import matplotlib.pyplot as plt
import numpy as np

plt.plot(X_cascade, np.array(Rate_Cascade)/np.array(X_cascade), label = "CascadeLake PSMN")
plt.ylim(0,1e6)
plt.xlim(1,100)
plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.savefig("sedov_scalling.svg")