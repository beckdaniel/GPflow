import tensorflow as tf
from param import Parameterized, Param


class WarpingFunction(Parameterized):
    """
    Abstract class that implements warping functions.
    """
    def __init__(self):
        Parameterized.__init__(self)
        self.num_gauss_hermite_points = 20

    def f(self, y):
        """
        The actual warping function
        """
        raise NotImplementedError

    def f_inv(self, z):
        """
        The inverse of the warping function
        """
        raise NotImplementedError


class IdentityFunction(WarpingFunction):
    """
    Identity warping function. This is for testing and sanity check purposes
    and should not be used in practice.
    """
    def __init__(self):
        WarpingFunction.__init__(self)

    def f(self, y):
        return tf.identity(y)

    def f_inv(self, z, y=None):
        return tf.identity(z)
