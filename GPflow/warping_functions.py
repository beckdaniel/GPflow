import tensorflow as tf
import numpy as np
from param import Parameterized, Param
import transforms


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


class LogFunction(WarpingFunction):
    """
    Easy wrapper for applying a fixed log warping function to
    positive-only values.
    """
    def __init__(self):
        WarpingFunction.__init__(self)

    def f(self, y):
        return tf.log(y)

    def f_inv(self, z, y=None):
        return tf.exp(z)


class TanhFunction(WarpingFunction):
    """
    This is the function proposed in Snelson et al (2004):
    A sum of tanh functions with linear trends outside
    the range. Notice the term 'd', which scales the
    linear trend.
    """
    def __init__(self, n_terms=3, initial_y=None):
        WarpingFunction.__init__(self)
        self.n_terms = n_terms
        self.num_parameters = 3 * self.n_terms + 1
        self.a = Param(np.ones((1, self.n_terms)), transforms.positive)
        self.b = Param(np.ones((1, self.n_terms)), transforms.positive)
        self.c = Param(np.ones((1, self.n_terms))) # no constraints needed
        self.d = Param(1.0, transforms.positive)
        self.initial_y = initial_y

    def f(self, y):
        """
        The first two terms replicate y and c to build two
        matrices of the same shape. These matrices are summed and
        then passed through the remaining terms of the tanh function.
        The result is get by summing the columns of the tanh
        matrix and adding the linear trend term.
        """
        y_repl = tf.matmul(y, np.ones((1, self.n_terms)))
        c_repl = tf.matmul(np.ones(y.get_shape()), self.c)
        y_plus_c = tf.add(y_repl, c_repl)
        tanh_matrix = tf.mul(self.a, tf.tanh(tf.mul(self.b, y_plus_c)))
        summ = tf.reduce_sum(tanh_matrix, 1, keep_dims=True)
        prod = tf.mul(self.d, y)
        return tf.add(prod, summ)

    def f_inv(self, z, y=None):
        return tf.exp(z)
