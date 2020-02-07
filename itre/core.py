import numpy as np
import os
import numba as nb
from .metadynamics import Metadynamics
from .atlas import Atlas

class Itre(object):
    """docstring for Itre."""
    def __init__(self):
        super(Itre, self).__init__()
        self.__required_properties_list = ['colvars_file','heights_file','sigmas_file']
        self.__optional_properties = ['kT','stride','thetas_file','iterations','starting_height','wall_file']

        for el in self.__required_properties_list:
            object.__setattr__(self,'{}'.format(el),None)

        for el in self.__optional_properties:
            object.__setattr__(self,'{}'.format(el),None)

        self.__setattr__('stride',10)
        self.__setattr__('kT',1.0)
        self.__setattr__('beta',1.0)
        self.__setattr__('iterations',20)
        self.__setattr__('starting_height',1.0)
        self.__setattr__('has_matrix',False)
        self.__setattr__('has_thetas',False)
        self.__setattr__('use_numba',False)

    def from_dict(self,dict):
        for key in dict.keys():
            if key in self.__required_properties_list or key in self.__optional_properties:
                self.__setattr__('{}'.format(key),dict[key])

        if os.path.isfile(self.colvars_file):
            colvars = np.loadtxt(self.colvars_file)
        if os.path.isfile(self.sigmas_file):
            sigmas = np.loadtxt(self.sigmas_file)
            if len(colvars)!=len(sigmas): raise ValueError('Length of colvars and sigmas is different!')
        if os.path.isfile(self.heights_file):
            heights = np.loadtxt(self.heights_file)
            if len(colvars)!=len(heights): raise ValueError('Length of colvars and heights is different!')
            heights /=heights[0]
            heights *= self.starting_height

        if self.thetas_file is not None:
            if os.path.isfile(self.thetas_file):
                thetas = np.loadtxt(self.thetas_file)
                if len(colvars)!=len(thetas): raise ValueError('Length of colvars and thetas is different!')
                self.__setattr__('has_thetas',True)

        if self.wall_file is not None:
            if os.path.isfile(self.wall_file):
                wall = np.loadtxt(self.wall_file)
                if len(colvars)!=len(wall): raise ValueError('Length of colvars and wall is different!')
        else:
            wall = np.zeros(len(colvars))

        self.__setattr__('colvars',colvars)
        self.__setattr__('wall',wall)
        self.__setattr__('sigmas',sigmas)
        self.__setattr__('heights',heights)
        self.__setattr__('beta',self.kT)

        if self.has_thetas:
            self.__setattr__('thetas',thetas)

        self.__setattr__('steps',len(self.colvars))
        self.__setattr__('n_evals',int(self.steps//self.stride))

    def calculate_bias_matrix(self):
        if self.has_thetas:
            bias_scheme = Atlas()
            if self.use_numba:
                matrix = bias_scheme.calculate_bias_matrix_nb(self.colvars,self.sigmas,self.heights,self.wall,self.thetas,self.n_evals,self.stride,len(self.colvars[0]))
            else:
                matrix = bias_scheme.calculate_bias_matrix(self.colvars,self.sigmas,self.heights,self.wall,self.thetas,self.n_evals,self.stride)
        else:
            bias_scheme = Metadynamics()
            if self.use_numba:
                matrix = bias_scheme.calculate_bias_matrix_nb(self.colvars,self.sigmas,self.heights,self.wall,self.n_evals,self.stride,len(self.colvars[0]))
            else:
                matrix = bias_scheme.calculate_bias_matrix(self.colvars,self.sigmas,self.heights,self.wall,self.n_evals,self.stride)
        self.has_matrix=True
        return matrix

    def calculate_c_t(self):
        if not self.has_matrix:
            self.bias_matrix=self.calculate_bias_matrix()
            self.instantaneous_bias = np.diag(self.bias_matrix)

        iter = 0
        self.ct = np.zeros((self.iterations,self.n_evals))
        matw = np.tril(np.exp(-self.bias_matrix*self.beta))
        for iteration in range(1,self.iterations):
            offset = self.instantaneous_bias-self.ct[iteration-1]
            vec1 = np.exp(offset*self.beta)
            res = matw.dot(vec1)
            norm = np.cumsum(vec1)
            self.ct[iteration] = -self.kT*np.log(res/norm)


if __name__ == '__main__':
    print("A __main__ implementation is still missing \n please implement it or use it as a module.\n")
