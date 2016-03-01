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
        #self.Y_untransformed = tf.convert_to_tensor(Y.copy())
        #self.Y = self.warp.f(self.Y_untransformed)
        self.Y_tensor = tf.convert_to_tensor(Y.copy())
        self.num_gauss_hermite_points = 20
        self.median = False

    def build_likelihood(self):
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        #self.Y = self.warp.f(self.Y_untransformed)
        f = self.warp.f(self.Y_tensor)
        ll = multivariate_normal(f, m, L)
        ll = tf.Print(ll, [ll], message="LL from WGP: ")
        #jacobian, = tf.gradients(f, self.Y_untransformed)
        #Y_tensor = tf.convert_to_tensor(self.Y)
        jacobian, = tf.gradients(f, self.Y_tensor)
        jacobian = tf.Print(jacobian, [jacobian], message="Jacobian: ")
        jacobian = tf.Print(jacobian, [self.Y_tensor], message="Y_tensor: ")
        w_ll = tf.add(ll, tf.reduce_sum(tf.log(jacobian)))
        w_ll = tf.Print(w_ll, [w_ll], message="Warped LL: ")
        return w_ll
        #return ll

    def build_predict(self, Xnew):
        Kd = self.kern.Kdiag(Xnew)
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.user_ops.triangular_solve(L, Kx, 'lower')
        f = self.warp.f(self.Y_tensor)
        #f = self.Y_tensor
        #V = tf.user_ops.triangular_solve(L, self.Y - self.mean_function(self.X), 'lower')
        V = tf.user_ops.triangular_solve(L, f - self.mean_function(self.X), 'lower')
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        fvar = Kd - tf.reduce_sum(tf.square(A), reduction_indices=0)
        return fmean, tf.tile(tf.reshape(fvar, (-1,1)), [1, self.Y.shape[1]])

    @AutoFlow(tf.placeholder(tf.float64))
    def predict_y(self, Xnew, pred_init=1):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        pred_f_mean = tf.Print(pred_f_mean, [pred_f_mean], message="WGP build predict mean: ")
        mean, var = self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
        mean = tf.Print(mean, [mean, tf.exp(mean)], message="WGP likelihood mean: ")
        std = tf.sqrt(var)
        if self.median:
            wmean = self.warp.f_inv(mean)
        else:
            wmean = tf.transpose(self._get_warped_mean(mean, std, pred_init=pred_init))
        wvar = tf.transpose(self._get_warped_variance(mean, std, pred_init=pred_init))
        wmean = tf.Print(wmean, [wmean], message="WGP warped mean: ")
        return wmean, wvar

    def _get_warped_term(self, mean, std, gh_x, pred_init=None):
        arg1 = tf.matmul(gh_x, tf.transpose(std)) * np.sqrt(2.0)
        arg2 = tf.matmul(tf.ones_like(gh_x), tf.transpose(mean))
        return self.warp.f_inv(arg1 + arg2, y=pred_init)

    def _get_warped_mean(self, mean, std, pred_init=None):
        """
        Calculate the warped mean using Gauss-Hermite quadrature.
        """
        gh_x, gh_w = np.polynomial.hermite.hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x[:, None]
        gh_w = gh_w[None, :]
        return tf.matmul(gh_w, self._get_warped_term(mean, std, gh_x)) / np.sqrt(np.pi)

    def _get_warped_variance(self, mean, std, pred_init=None):
        """
        Calculate the warped variance using Gauss-Hermite quadrature.
        """
        gh_x, gh_w = np.polynomial.hermite.hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x[:,None]
        gh_w = gh_w[None,:]
        arg1 = tf.matmul(gh_w, tf.pow(self._get_warped_term(mean, std, gh_x, pred_init=pred_init), 2)) 
        arg1 = tf.div(arg1, np.sqrt(np.pi))
        arg2 = self._get_warped_mean(mean, std, pred_init=pred_init)
        return tf.sub(arg1, tf.pow(arg2, 2))
