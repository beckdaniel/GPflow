import tensorflow as tf
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

    def build_likelihood(self):
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        ll = multivariate_normal(self.Y, m, L)
        f = self.warp.f(self.Y_untransformed)
        print f
        jacobian, = tf.gradients(f, self.Y_untransformed)
        print jacobian
        #jacobian = self.warping_function.fgrad_y(self.Y_untransformed)
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
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
