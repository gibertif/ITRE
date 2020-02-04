import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import numba as nb

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def get_overlap(colvar,sigma,step):
    overlap = np.zeros((step,step))
    for i in range(step):
        cv1 = colvar[i]
        for j in range(i+1):
            cv2 = colvar[j]
            dist = (cv1-cv2)/sigma
            dist = 0.5 * dist.dot(dist)
            overlap[i,j] += np.exp(-dist)

    return overlap

def recursive_bias_matrix_2(colvar,sigma,height,start=0,end=10):
    steps=len(colvar)
    bias_matrix = np.zeros((steps-start,end-start))

    for i in range(start,end):
        for k in range(i):
            bias_matrix[i-start,i-start] += gaussian(colvar[i],colvar[k],sigma)*height[k]

    for i in range(start,end):
        for j in range(i+1,steps):
            bias_matrix[j-start,i-start] = bias_matrix[j-start-1,i-start] + gaussian(colvar[i],colvar[j-1],sigma)*height[j-1]

    return bias_matrix

def recursive_bias_matrix(colvar,sigma,height,start=0,end=10):
    bias_matrix = np.zeros((end-start,end-start))

    for i in range(start,end):
        for k in range(i):
            bias_matrix[i-start,i-start] += gaussian(colvar[i],colvar[k],sigma)*height[k]

    for i in range(start,end):
        for j in range(i+1,end):
            bias_matrix[j-start,i-start] = bias_matrix[j-start-1,i-start] + gaussian(colvar[i],colvar[j-1],sigma)*height[j-1]

    return bias_matrix

@nb.jit
def get_whole_matrix(colvar,sigma,height,steps):
    bias_matrix = np.zeros((steps,steps))

    for i in range(steps):
        for j in range(i+1):
            cv1 = colvar[j]
            for step in range(i):
                cv2 = colvar[step]
                dist = (cv1-cv2)/sigma
                dist = 0.5 * dist.dot(dist)
                bias_matrix[i,j] += np.exp(-dist)*height[step]

    return bias_matrix

def gaussian(a,b,s):
    dist = (a-b)/s
    dist = 0.5 * dist.dot(dist)
    return np.exp(-dist)

def get_ct_2(bias_matrix,bias,prev_ct=0):
    iter = 0
    t = len(bias)
    t2=bias_matrix.shape[0]
    ct = np.zeros(t2)
    matw = np.tril(np.exp(-bias_matrix))

    if isinstance(prev_ct,np.ndarray) and prev_ct.size != 0:
        t1=len(prev_ct)
        final_ct = np.exp(-prev_ct[-1])
        prev_norm = np.exp(bias[:t1]-prev_ct).sum()
    else:
        final_ct = 0
        prev_norm = 0

    while iter < 10:
        offset = bias-ct
        vec1 = np.exp(offset)
        res = matw.T.dot(vec1)+prev_norm*final_ct
        norm = np.cumsum(vec1)+prev_norm
        print(res.shape,norm.shape)
        ct = -np.log(res/norm)
        iter = iter + 1

    return ct

def plot_matrix(matrix):
    plt.matshow(matrix)
    plt.colorbar()

def plot_multi_line(llist):
    for el in llist:
        plt.plot(el)
    plt.show()


dr = '../plain_meta/'

dr = '../wt_meta/'
hills = np.loadtxt('{}/HILLS'.format(dr))
steps = len(hills)
steps = 500
colvar = hills[:,1:3]
sigma = hills[0,3:5]

if 'wt' in dr:
    height = hills[:,5]
    height *=  2.0/height[0]
    bias = np.loadtxt('{}/rr.colvar_500'.format(dr),usecols=3,skiprows=2)
else:
    height = hills[:,5]
    bias = np.loadtxt('{}/bias'.format(dr),usecols=3,skiprows=2)


bias_matrix_1 = get_whole_matrix(colvar,sigma,height,steps)
c_t = get_ct(bias_matrix_1,bias[:steps])

logw = np.loadtxt('../wt_meta/logw')
steps=3000
logw = logw.reshape(3000,3000)
c_t = get_ct(logw,bias[:3000])

def get_ct(bias_matrix,bias,prev_ct=0):
    iter = 0
    t = len(bias)
    t2=len(bias_matrix)
    ct = np.zeros(t2)
    matw = np.tril(np.exp(-bias_matrix))
    if isinstance(prev_ct,np.ndarray) and prev_ct.size != 0:
        t1=len(prev_ct)
        final_ct = np.exp(-prev_ct[-1])
        offset = bias[:t1]-prev_ct
        prev_norm = np.exp(offset).sum()
    else:
        t1=0
        final_ct = 0
        prev_norm = 0

    while iter < 20:
        offset = bias[t1:t2+t1]-ct
        vec1 = np.exp(offset)
        res = matw.dot(vec1) + prev_norm*final_ct
        norm = np.cumsum(vec1) + prev_norm
        ct = -np.log(res/norm)
        iter = iter + 1

    return ct

plt.plot(logw[100,:100]/logw[2500,:100])


plt.figure(1)
plt.title('c(t)')
plt.figure(2)
plt.title('V(s(t),t)-c(t)')
plt.figure(3)
plt.title('exp[V(s(t),t)-c(t)]')
plt.figure(4)
plt.title(r'c(t)-c$_i$(t)')

plt.figure(1)
plt.plot(bias[:steps],alpha=0.5)
plt.plot(c_t)
plt.figure(2)
plt.plot(bias[:steps]-c_t-np.amax(bias[:steps]-c_t))
plt.figure(3)
weights=np.exp(bias[:steps]-c_t-np.amax(bias[:steps]-c_t))
plt.plot(weights)
ct=np.zeros(steps)
cc=int(steps/500)
for i in range(cc):
    start=int(steps*i/cc) ; end=int(steps*(i+1)/cc)
    mat = recursive_bias_matrix(colvar[:steps],sigma,height,start=start,end=end)
    ct[start:end]=get_ct(logw[start:end,start:end],bias[:end],prev_ct=ct[:start])
    plt.figure(1)
    plt.plot(np.arange(start,end),ct[start:end],'--')
    plt.figure(2)
    plt.plot(np.arange(start,end),bias[start:end]-ct[start:end]-np.amax(bias[start:end]-ct[start:end]),alpha=0.7)
    plt.figure(3)
    weights=np.exp(bias[start:end]-ct[start:end]-np.amax(bias[start:end]-ct[start:end]))
    plt.plot(np.arange(start,end),weights,alpha=0.7)
    plt.figure(4)
    plt.plot(np.arange(start,end),c_t[start:end]-ct[start:end],alpha=0.7)












import sympy as sp

nn=steps
matrix = np.zeros((nn,nn),dtype=str)
matrix = matrix.tolist()

overlap = np.zeros((nn,nn),dtype=str)
overlap = overlap.tolist()
hhh = []

for  i in range(nn):
    hhh.append('h({})'.format(i))
    for j in range(i+1):
        overlap[i][j] = ' g({};{}) + '.format(j,i)
        for step in range(i):
            matrix[i][j] = matrix[i][j] + ' g({};{})h({}) + '.format(j,step,step)
        #matrix[i][j] = matrix[i][j] + ' g({};{})h({}) '.format(j,i,i)
    for j in range(i+1,nn):
        matrix[i][j]='0'


for l in np.diag(matrix):
    print(l)

hhh = np.array(hhh)

hhh
overlap = np.array(overlap)
overlap

matrix = np.array(matrix)
import pandas as pd

df = pd.DataFrame(matrix)
df.to_csv("pot_matrix.csv")

if __name__ == '__main__':

    import json
    import os

    if os.path.isfile('pyitre.json'):
        data = json.loads('pyitre.json')





    # import argparse as ap
    #
    # parser = ap.ArgumentParser(description='Implementation of ITRE in a python script that can reweight METAD and ATLAS calculations. Useful to play with but non sufficient for long calculations.')
    # parser.add_argument('--colvar','-c',type=str,'File containing the colvar file.')
    # parser.add_argument('--height','-h',type=str,'File containing height of the hills.')
    # parser.add_argument('--theta','-t',type=str,'File containing the assignation to a basin for a ATLAS calculation.')
    # parser.add_argument('--kbt','-k',type=int,default=1,'KbT in the correct units to run the reweighing.')
    # parser.add_argument('--stride','-s',type=int,default=10,'Stride used in the evaluation of the c(t).')
    #
    # args = parser.parse_args()
    #
    # colvar_file=args.colvar
    # height_file=args.height
    # stride=args.stride
    # kbt=args.kbt
    #
    # theta_file=args.theta
    #
    # colvar=np.loadtxt(colvar_file)
    # theta=np.loadtxt(theta_file)
    # height=np.loadtxt(height_file)
    #
    # if bool_is_atlas:
    #     potential_matrix = calculate_potential_matrix(colvar,theta,height)
    #
    # else:
