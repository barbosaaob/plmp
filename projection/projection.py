"""
Abstract projection technique.

This package implements an abstract class for projection techniques used in
visualization area. It requires NumPy (http://www.numpy.org/) and matplotlib
(http://www.matplotlib.org/) libraries.
"""
from __future__ import print_function

try:
    import numpy as np
    import matplotlib.pyplot as mpl
except ImportError as msg:
    error = ", please install the following packages:\n"
    error += "    NumPy      (http://www.numpy.org)\n"
    error += "    matplotlib (http://www.matplotlib.org)"
    raise ImportError(str(msg) + error)


class Projection():
    """
    Abstract projection class.

    This class implements an abstract projection technique.
    """
    def __init__(self, data, data_class, projection_dim=2):
        """
        Class constructor.

        data = numpy.ndarray type object. Data, each line means a data
        instance.
        data_class = numpy.ndarray type object. Data class.
        projection_dim = int type object. Visual space dimension (optional,
        default is 2 (R^2)).
        """
        # data
        self.data = data
        self.data_class = data_class
        self.data_ninstances = data.shape[0]
        self.data_dim = data.shape[1]
        # projection
        self.projection = np.zeros((self.data_ninstances, projection_dim))
        self.projection_dim = projection_dim

    def __str__(self):
        """
        To string method.

        Returns a string with information about the projection.
        """
        s = "Projection info:\n"
        s += "    #instances: " + str(self.data_ninstances) + "\n"
        s += "    data dimension:  " + str(self.data_dim) + "\n"
        s += "    projection dimension:  " + str(self.projection_dim) + "\n"
        s += "    data: " + str(self.data[0]) + "\n"
        s += "          " + str(self.data[1]) + "...\n"
        s += "    projection: " + str(self.projection[0]) + "\n"
        s += "          " + str(self.projection[1]) + "..."
        return s

    def get_data(self):
        """
        Get data.

        Returns the data stored in the object.
        """
        return self.data

    def get_data_class(self):
        """
        Get data class.

        Returns the data class stored in the object.
        """
        return self.data_class

    def get_data_dim(self):
        """
        Get data dimension.

        Returns the data dimension stored in the object.
        """
        return self.data_dim

    def get_data_ninstances(self):
        """
        Get data number of instances.

        Returns the number of instances in the data stored in the object.
        """
        return self.data_ninstances

    def get_projection(self):
        """
        Get projection.

        Returns the projection stored in the object.
        """
        return self.projection

    def plot(self):
        """
        Plot the projection.

        Plot the projection computed by a technique.
        """
        y = self.projection
        mpl.scatter(y[:, 0], y[:, 1], c=self.data_class)
        mpl.show()


def run():
    """
    Run function.
    """
    print("This is an bstract class.")
    print("You have to implement a projection technique.")

if __name__ == "__main__":
    run()
