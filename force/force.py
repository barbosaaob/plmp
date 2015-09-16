"""
Force

Force multidimensional projection technique.
http://www.lcad.icmc.usp.br/%7Enonato/pubs/TejadaEtAl.pdf
"""

from __future__ import print_function
from projection import projection

try:
    import numpy as np
    import scipy.spatial.distance as dist
except ImportError as msg:
    error = ", please install the following packages:\n"
    error += "    NumPy      (http://www.numpy.org)\n"
    error += "    SciPy      (http://www.scipy.org)"
    raise ImportError(str(msg) + error)


class Force(projection.Projection):
    """
    Force projection.
    """
    def __init__(self, data, data_class, dtype="data", delta_frac=8,
                 niter=50, tol=1.0e-6):
        """
        Class initialization.
        """
        projection.Projection.__init__(self, data, data_class, 2)
        self.dtype = dtype
        self.delta_frac = delta_frac
        self.niter = niter
        self.tol = tol

    def project(self):
        """
        Project method.

        Projection itself.
        """
        assert type(self.data) is np.ndarray, \
            "*** ERROR (Force): project input must be of numpy.ndarray \
                type."

        # number of instances, dimension of the data
        ninst = self.data_ninstances

        # random initialization
        Y = np.random.random((ninst, self.projection_dim))

        # computes distance in R^n
        if self.dtype == "data":
            distRn = dist.squareform(dist.pdist(self.data))
        elif self.dtype == "dmat":
            distRn = self.data
        else:
            print("*** ERROR (Force): Undefined data type.")
        assert type(distRn) is np.ndarray and distRn.shape == (ninst, ninst), \
            "*** ERROR (Force): project input must be numpy.ndarray \
                type."

        idx = np.random.permutation(ninst)

        for k in range(self.niter):
            # for each x'
            for i in range(ninst):
                inst1 = idx[i]
                # for each q' != x'
                for j in range(ninst):
                    inst2 = idx[j]
                    if inst1 != inst2:
                        # computes direction v
                        v = Y[inst2] - Y[inst1]
                        distR2 = np.hypot(v[0], v[1])
                        if distR2 < self.tol:
                            distR2 = self.tol
                        delta = (distRn[inst1][inst2] - distR2)/self.delta_frac
                        v /= distR2
                        # move q' = Y[j] in the direction of v by a fraction
                        # of delta
                        Y[inst2] += delta * v
        self.projection = Y


def run():
    import time
    import sys
    print("Loading data set... ", end="")
    sys.stdout.flush()
    data_file = np.loadtxt("iris.data")
    print("Done.")
    n, dim = data_file.shape
    data = data_file[:, range(dim - 1)]
    data_class = data_file[:, dim - 1]
    start_time = time.time()
    print("Projecting... ", end="")
    sys.stdout.flush()
    f = Force(data, data_class)
    f.project()
    print("Done. (" + str(time.time() - start_time) + "s)")
    f.plot()

if __name__ == "__main__":
    run()
