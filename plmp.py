"""
PLMP

PLMP multidimensional projection technique.
http://www.lcad.icmc.usp.br/%7Enonato/pubs/plmp.pdf
"""
from __future__ import print_function
from projection import projection
from force import force

try:
    import numpy as np
except ImportError as msg:
    error = ", please install the following packages:\n"
    error += "    NumPy      (http://www.numpy.org)\n"
    raise ImportError(str(msg) + error)


class PLMP(projection.Projection):
    """
    PLMP projection.
    """
    def __init__(self, data, data_class, sample=None, sample_projection=None):
        """
        Class initialization.
        """
        assert type(data) is np.ndarray, "*** ERROR (PLMP): Data is of wrong \
                type!"

        projection.Projection.__init__(self, data, data_class)
        self.sample = sample
        self.sample_projection = sample_projection

    def project(self, tol=1e-6):
        """
        Projection method.

        Projection itself.
        """
        ninst, dim = self.data.shape    # number os instances, data dimension
        p = self.projection_dim         # visual space dimension
        x = self.data
        xs = self.data[self.sample, :]
        ys = self.sample_projection

        Phi = np.zeros((dim, p))
        L = xs.T

        for j in range(p):
            Phi[:, j] = ys[:, j].T.dot(L.T.dot(np.linalg.inv(L.dot(L.T))))

        for i in range(ninst):
            self.projection[i, :] = x[i, :].dot(Phi)


def run():
    import time
    import sys
    print("Loading data set... ", end="")
    sys.stdout.flush()
    data_file = np.loadtxt("iris.data")
    print("Done.")
    ninst, dim = data_file.shape
    sample_size = int(np.ceil(np.sqrt(ninst)))
    data = data_file[:, range(dim - 1)]
    data_class = data_file[:, dim - 1]
    sample = np.random.permutation(ninst)
    sample = sample[range(sample_size)]

    # force
    start_time = time.time()
    print("Projecting samples... ", end="")
    sys.stdout.flush()
    f = force.Force(data[sample, :], [])
    f.project()
    sample_projection = f.get_projection()
    print("Done. (" + str(time.time() - start_time) + "s.)")

    # PLMP
    start_time = time.time()
    print("Projecting... ", end="")
    sys.stdout.flush()
    plmp = PLMP(data, data_class, sample, sample_projection)
    plmp.project()
    print("Done. (" + str(time.time() - start_time) + "s.)")
    plmp.plot()


if __name__ == "__main__":
    run()
