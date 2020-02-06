import numpy as np
import numba as nb

class Atlas(object):
    """docstring for Atlas."""

    def __init__(self):
        super(Atlas, self).__init__()

    def kernel(self,a,b,c):
        dist = (a-b)/c
        dist = 0.5 * dist.dot(dist)
        return np.exp(-dist)

    def calculate_bias_matrix(self,colvars,sigmas,heights,wall,thetas,n_evals,stride):
        n_minima = len(thetas[0])
        dims = int(len(colvars[0])//n_minima)
        bias_matrix = np.zeros((n_evals,n_evals))

        for i in range(n_evals):
            upper_index = int(i*stride)
            for k in range(upper_index+1):
                sum_bias = 0.0
                for minimum in range(n_minima):
                    start = int(minimum*dims) ; end = int(minimum*dims)+dims
                    switch = thetas[i,minimum]*thetas[k,minimum]
                    sum_bias += self.kernel(colvars[upper_index,start:end],colvars[k,start:end],sigmas[k,start:end])*heights[k]*switch

                bias_matrix[i,i] += sum_bias + wall[k]

        for i in range(n_evals):
            ref_index= int(i*stride)
            for j in range(i,n_evals-1):
                lower_index = int(j*stride)
                upper_index = int((j+1)*stride)
                bias_sum = 0.0
                for t in range(lower_index,upper_index):
                    for minimum in range(n_minima):
                        start = int(minimum*dims) ; end = int(minimum*dims)+dims
                        switch = thetas[ref_index,minimum]*thetas[t,minimum]
                        sum_bias += self.kernel(colvars[ref_index,start:end],colvars[t,start:end],sigmas[t,start:end])*heights[t]*switch

                    sum_bias += wall[t]

                bias_matrix[j+1,i] = bias_matrix[j,i] + bias_sum

        return bias_matrix

    @staticmethod
    @nb.jit
    def calculate_bias_matrix_nb(colvars,sigmas,heights,wall,thetas,n_evals,stride,dims):
        n_minima = len(thetas[0])
        dims = int(len(colvars[0])//n_minima)
        bias_matrix = np.zeros((n_evals,n_evals))
        dist = np.zeros(dims)
        dist2 = np.zeros(dims)

        for i in range(n_evals):
            for k in range(i*stride+1):
                sum_bias = 0.0
                for minimum in range(n_minima):
                    start = int(minimum*dims) ; end = int(minimum*dims)+dims
                    dist = (colvars[i*stride,start:end]-colvars[k,start:end])/sigmas[k,start:end]
                    dist2 = 0.5 * dist.dot(dist)
                    sum_bias += np.exp(-dist2)*heights[k]*thetas[i,minimum]*thetas[k,minimum]

                bias_matrix[i,i] += sum_bias + wall[k]

        for i in range(n_evals):
            for j in range(i,n_evals-1):
                bias_sum = 0.0
                for t in range(j*stride,(j+1)*stride):
                    for minimum in range(n_minima):
                        start = int(minimum*dims) ; end = int(minimum*dims)+dims
                        dist = (colvars[i*stride,start:end]-colvars[t,start:end])/sigmas[t,start:end]
                        dist2 = 0.5 * dist.dot(dist)
                        sum_bias += np.exp(-dist2)*heights[t]*thetas[i*stride,minimum]*thetas[t,minimum]

                    sum_bias += wall[t]

                bias_matrix[j+1,i] = bias_matrix[j,i] + bias_sum

        return bias_matrix
