import tensorflow as tf
import numpy as np
from .model import GPModel, AutoFlow
from .param import Param
from .mean_functions import Zero
from .likelihoods import Gaussian
from tf_hacks import eye
from .densities import multivariate_normal


class WarpedGP(GPModel):
    """
    Warped Gaussian Processes (Snelson et al. 2004).

    It is a standard GP where the latent values are
    warped observations. It requires a warping function,
    which should be a monotonic function.
    """
    def __init__(self, X, Y, kernel, warp, mean_function=Zero()):
        likelihood = Gaussian()
        GPModel.__init__(self, X, Y, kernel, likelihood, mean_function)
        self.warp = warp
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.Y_untransformed = tf.convert_to_tensor(Y.copy())
        self.num_gauss_hermite_points = 20

    def build_likelihood(self):
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        ll = multivariate_normal(self.Y, m, L)
        f = self.warp.f(self.Y_untransformed)
        jacobian, = tf.gradients(f, self.Y_untransformed)
        return tf.add(ll, tf.reduce_sum(tf.log(jacobian)))

    def build_predict(self, Xnew):
        Kd = self.kern.Kdiag(Xnew)
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.user_ops.triangular_solve(L, Kx, 'lower')
        V = tf.user_ops.triangular_solve(L, self.Y - self.mean_function(self.X), 'lower')
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        fvar = Kd - tf.reduce_sum(tf.square(A), reduction_indices=0)
        return fmean, tf.tile(tf.reshape(fvar, (-1,1)), [1, self.Y.shape[1]])

    @AutoFlow(tf.placeholder(tf.float64))
    def predict_y(self, Xnew, pred_init=1):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        mean, var = self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
        std = tf.sqrt(var)
        wmean = self._get_warped_mean(mean, std, pred_init=pred_init).T
        wvar = self._get_warped_variance(mean, std, pred_init=pred_init).T
        return wmean, wvar

    def _get_warped_term(self, mean, std, gh_x, pred_init=None):
        arg1 = gh_x.dot(std.T) * np.sqrt(2)
        arg2 = np.ones(shape=gh_x.shape).dot(mean.T)
        return self.warping_function.f_inv(arg1 + arg2, y=pred_init)

    def _get_warped_mean(self, mean, std, pred_init=None):
        """
        Calculate the warped mean by using Gauss-Hermite quadrature.
        """
        gh_x, gh_w = np.polynomial.hermite.hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x[:, None]
        gh_w = gh_w[None, :]
        return gh_w.dot(self._get_warped_term(mean, std, gh_x)) / np.sqrt(np.pi)

    def _get_warped_variance(self, mean, std, pred_init=None):
        """
        Calculate the warped variance by using Gauss-Hermite quadrature.
        """
        gh_x, gh_w = np.polynomial.hermite.hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x[:,None]
        gh_w = gh_w[None,:]
        arg1 = gh_w.dot(self._get_warped_term(mean, std, gh_x, pred_init=pred_init) ** 2) 
        arg1 /= np.sqrt(np.pi)
        arg2 = self._get_warped_mean(mean, std, pred_init=pred_init)
        return arg1 - (arg2 ** 2)
