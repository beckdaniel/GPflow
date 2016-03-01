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
        The first term replicate y into a tensor of n_terms. The second one
        adds 'c' via broadcasting, then it passes through the remaining 
        terms of the tanh function.
        The result is get by summing the columns of the tanh
        tensor and adding the linear trend term.
        """
        y_repl = tf.matmul(y, np.ones((y.get_shape()[1], self.n_terms)))
        y_plus_c = tf.add(y_repl, self.c)
        tanh_tensor = tf.mul(self.a, tf.tanh(tf.mul(self.b, y_plus_c)))
        summ = tf.reduce_sum(tanh_tensor, 1, keep_dims=True)
        prod = tf.mul(self.d, y)
        return tf.add(prod, summ)

    def f_inv(self, z, max_its=50, y=None):
        """
        No closed form the inverse is available so we use a first
        order Newton method here.
        """
        y = tf.transpose(tf.ones_like(z))
        it = 0
        update_sum = tf.Variable(tf.constant(0, tf.float64))
        zt = tf.transpose(z)
        rate = np.array([0.9])
        while (update_sum > 10 and it < max_its):
            fy = self.f(y)
            fgrady = tf.gradients(fy, y)[0]
            #fgrady2 = tf.gradients(fgrady, y)[0]
            sub_term = tf.sub(fy, zt)
            #sec_order_term = tf.div(tf.mul(sub_term, fgrady2), tf.mul(np.array([2.0]), fgrady))
            #update = tf.div(sub_term, tf.sub(fgrady, sec_order_term))
            update = tf.div(sub_term, fgrady)
            #update = tf.Print(update, [tf.reduce_sum(update), y], message="Update in f_inv: ")
            y = tf.sub(y, tf.mul((rate ** it), update))
            it += 1
            print it
            update_sum.assign(tf.reduce_sum(update))
        if it == max_its:
            print("WARNING!!! Maximum number of iterations reached in f_inv ")
            y = tf.Print(y, [tf.reduce_sum(update)], message="Total update in f_inv: ")
            #print("Sum of updates: %.4f" % tf.reduce_sum(update))
        return tf.transpose(y)
