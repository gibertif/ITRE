This modules contains a python code that can be used to calculate c(t) starting form a metadynamics or an atlas calculation.

There are minimalist examples in the **examples** folder that illustrate how to use the module, as well as the *clean_files.py* scripts in the tool directory.

The code can be used with *plain* python, but is not very fast, or with *numba*, for which there is a considerable speed up.

I am planning of implementing a version with *mpi4py* too to further increase the scaling of the method.

Keep an eye on this!
