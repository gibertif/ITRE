import sys
sys.path.append('../../')
import itre
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import time

if os.path.isfile("pyitre.json"):
    with open("pyitre.json",'r') as json_file:
        directives = json.load(json_file)


i_bias = np.loadtxt('bias')

it = itre.Itre()
it.from_dict(directives)
it.use_numba=True
start=time.time()
it.calculate_c_t()
end=time.time()
raw_time=end-start
ref_ct = it.ct[-1].T
plt.plot(ref_ct,'--o')
plt.show()

plt.plot(i_bias)
plt.plot(it.instantaneous_bias,alpha=0.6)
plt.plot(it.thetas)
plt.show()

plt.plot(it.instantaneous_bias-i_bias,alpha=0.6)
plt.show()

# plt.matshow(it.bias_matrix)
# plt.show()

#colvars=np.loadtxt('COLVARS')
#sigmas=np.loadtxt('SIGMAS')
#heights=np.loadtxt('HEIGHTS')
#
#times=[]
#for ss in range(1000,len(colvars),200):
#    new_it=itre.Itre()
#    new_it.use_numba=True
#    new_it.colvars=colvars[:ss]
#    new_it.sigmas=sigmas[:ss]
#    new_it.heights=heights[:ss]/heights[0]*2.0
#    new_it.stride=10
#    new_it.n_evals=int(ss/10)
#    start = time.time()
#    new_it.calculate_c_t()
#    end = time.time()
#    plt.plot(new_it.ct[-1].T)
#    times.append(end-start)
#
#plt.show()
#
#plt.plot(raw_time/np.array(times))
#plt.show()
#
#print(raw_time/times)
