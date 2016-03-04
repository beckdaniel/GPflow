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
        self.Y_original = tf.convert_to_tensor(Y.copy())
        self.num_gauss_hermite_points = 20
        self.median = False

    def build_likelihood(self):
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        f = self.warp.f(self.Y_original)
        ll = multivariate_normal(f, m, L)
        jacobian = tf.gradients(f, self.Y_original)[0] # gradient returns a list
        return tf.add(ll, tf.reduce_sum(tf.log(jacobian)))

    def build_predict(self, Xnew):
        Kd = self.kern.Kdiag(Xnew)
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.user_ops.triangular_solve(L, Kx, 'lower')
        f = self.warp.f(self.Y_original)
        V = tf.user_ops.triangular_solve(L, f - self.mean_function(self.X), 'lower')
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        fvar = Kd - tf.reduce_sum(tf.square(A), reduction_indices=0)
        return fmean, tf.tile(tf.reshape(fvar, (-1,1)), [1, self.Y.shape[1]])

    @AutoFlow(tf.placeholder(tf.float64))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew.
        If the median flag is set, compute the median instead of the mean.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        mean, var = self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
        std = tf.sqrt(var)
        if self.median:
            wmean = tf.transpose(self.warp.f_inv(tf.transpose(mean)))
            _, wvar = self._get_warped_mean_and_variance(mean, std)
            #wvar = var
        else:
            wmean, wvar = self._get_warped_mean_and_variance(mean, std)
        
        return wmean, wvar

    @AutoFlow(tf.placeholder(tf.float64), tf.placeholder(tf.float64))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log denisty of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        f_new = self.warp.f(Ynew)
        ll_new = self.likelihood.predict_density(pred_f_mean, pred_f_var, f_new)
        jacobian_new = tf.gradients(f_new, Ynew)[0] # gradient returns a list
        return ll_new + tf.log(jacobian_new)

    def _get_warped_term(self, mean, std, gh_x):
        arg1 = tf.matmul(gh_x, tf.transpose(std)) * np.sqrt(2.0)
        arg2 = tf.matmul(tf.ones_like(gh_x), tf.transpose(mean))
        return self.warp.f_inv(tf.add(arg1, arg2))

    def _get_warped_mean(self, mean, std):
        """
        Calculate the warped mean using Gauss-Hermite quadrature.
        """
        gh_x, gh_w = np.polynomial.hermite.hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x[:, None]
        gh_w = gh_w[None, :]
        return tf.matmul(gh_w, self._get_warped_term(mean, std, gh_x)) / np.sqrt(np.pi)

    def _get_warped_variance(self, mean, std):
        """
        Calculate the warped variance using Gauss-Hermite quadrature.
        """
        gh_x, gh_w = np.polynomial.hermite.hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x[:, None]
        gh_w = gh_w[None, :]
        arg1 = tf.matmul(gh_w, tf.pow(self._get_warped_term(mean, std, gh_x), 2)) 
        arg1 = tf.div(arg1, np.sqrt(np.pi))
        arg2 = self._get_warped_mean(mean, std)
        return tf.sub(arg1, tf.pow(arg2, 2))

    def _get_warped_mean_and_variance(self, mean, std):
        """
        This is a shortcut method when both mean and variance are needed.
        It calls the inverse warping function only once, so it is faster.
        """
        gh_x, gh_w = np.polynomial.hermite.hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x[:, None]
        gh_w = gh_w[None, :]
        warped_term = self._get_warped_term(mean, std, gh_x)
        warped_mean = tf.matmul(gh_w, warped_term) / np.sqrt(np.pi)
        arg1 = tf.div(tf.matmul(gh_w, tf.pow(warped_term, 2)), np.sqrt(np.pi))
        warped_var = tf.sub(arg1, tf.pow(mean, 2))
        return tf.transpose(warped_mean), tf.transpose(warped_var)
