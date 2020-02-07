import numpy as np
import numba as nb


class Metadynamics(object):
    """docstring for Metadynamics."""

    def __init__(self):
        super(Metadynamics, self).__init__()
        pass

    def kernel(self,a,b,c):
        dist = (a-b)/c
        dist = 0.5 * dist.dot(dist)
        return np.exp(-dist)

    @staticmethod
    @nb.jit
    def calculate_bias_matrix_nb(colvars,sigmas,heights,wall,n_evals,stride,dims):
        bias_matrix = np.zeros((n_evals,n_evals))
        dist = np.zeros(dims)

        for i in range(n_evals):
            for k in range(i*stride):
                dist = (colvars[i*stride]-colvars[k])/sigmas[k]
                dist2 = 0.5 * dist.dot(dist)
                bias_matrix[i,i] += np.exp(-dist2)*heights[k] + wall[k]

        for i in range(n_evals):
            for j in range(i,n_evals-1):
                bias_sum = 0.0
                for t in range(j*stride,(j+1)*stride):
                    dist = (colvars[i*stride]-colvars[t])/sigmas[t]
                    dist2 = 0.5 * dist.dot(dist)
                    bias_sum += np.exp(-dist2)*heights[t] + wall[t]

                bias_matrix[j+1,i] = bias_matrix[j,i] + bias_sum

        return bias_matrix

    def calculate_bias_matrix(self,colvars,sigmas,heights,wall,n_evals,stride):
        bias_matrix = np.zeros((n_evals,n_evals))

        for i in range(n_evals):
            upper_index = int(i*stride)
            for k in range(upper_index):
                bias_matrix[i,i] += self.kernel(colvars[upper_index],colvars[k],sigmas[k])*heights[k] + wall[k]

        for i in range(n_evals):
            ref_index= int(i*stride)
            for j in range(i,n_evals-1):
                lower_index = int(j*stride)
                upper_index = int((j+1)*stride)
                bias_sum = 0.0
                for t in range(lower_index,upper_index):
                    bias_sum += self.kernel(colvars[ref_index],colvars[t],sigmas[t])*heights[t]  + wall[t]

                bias_matrix[j+1,i] = bias_matrix[j,i] + bias_sum

        return bias_matrix
