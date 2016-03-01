import GPflow
import tensorflow as tf
import numpy as np
import unittest

@unittest.skip('')
class TestMethods(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(100,2)
        self.Y = self.rng.randn(100,1)
        self.Z = self.rng.randn(10,2)
        self.lik = GPflow.likelihoods.Gaussian()
        self.kern = GPflow.kernels.Matern32(2)
        self.Xs = self.rng.randn(10,2)

        #make one of each model
        self.ms = []
        for M in (GPflow.vgp.VGP, GPflow.gpmc.GPMC):
            self.ms.append( M(self.X, self.Y, self.kern, self.lik) )
        for M in (GPflow.sgpmc.SGPMC, GPflow.svgp.SVGP):
            self.ms.append( M(self.X, self.Y, self.kern, self.lik, self.Z) )
        self.ms.append(GPflow.gpr.GPR(self.X, self.Y, self.kern))


    def test_sizes(self):
        for m in self.ms:
            m._compile()
            f,g = m._objective(m.get_free_state())
            self.failUnless(f.size == 1)
            self.failUnless(g.size == m.get_free_state().size)
        
    def test_prediction_f(self):
        for m in self.ms:
            m._compile()
            mf, vf = m.predict_f(self.Xs)
            self.failUnless(mf.shape == vf.shape)
            self.failUnless(mf.shape == (10, 1))
            self.failUnless(np.all(vf >= 0.0))

    def test_prediction_y(self):
        for m in self.ms:
            m._compile()
            mf, vf = m.predict_y(self.Xs)
            self.failUnless(mf.shape == vf.shape)
            self.failUnless(mf.shape == (10, 1))
            self.failUnless(np.all(vf >= 0.0))

    def test_prediction_density(self):
        self.Ys = self.rng.randn(10,1)
        for m in self.ms:
            m._compile()
            d = m.predict_density(self.Xs, self.Ys)
            self.failUnless(d.shape == (10, 1))


@unittest.skip('')
class TestSVGP(unittest.TestCase):
    """
    The SVGP has four modes of operation. with and without whitening, with and without diagonals.

    Here we make sure thet the bound on the likelihood is the same when using both representations (as far as possible)
    """
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(20, 1)
        self.Y = self.rng.randn(20,2)
        self.Z = self.rng.randn(3,1)

    def test_white(self):
        m1 = GPflow.svgp.SVGP(self.X, self.Y, kern=GPflow.kernels.RBF(1)+GPflow.kernels.White(1), likelihood=GPflow.likelihoods.Exponential(),
                           Z=self.Z, q_diag=True, whiten=True)
        m2 = GPflow.svgp.SVGP(self.X, self.Y, kern=GPflow.kernels.RBF(1)+GPflow.kernels.White(1), likelihood=GPflow.likelihoods.Exponential(),
                           Z=self.Z, q_diag=False, whiten=True)
        m1._compile()
        m2._compile()

        qsqrt, qmean = self.rng.randn(2, 3, 2)
        qsqrt = (qsqrt**2)*0.01
        m1.q_sqrt = qsqrt
        m1.q_mu = qmean
        m2.q_sqrt = np.array([np.diag(qsqrt[:,0]), np.diag(qsqrt[:,1])]).swapaxes(0,2)
        m2.q_mu = qmean
        self.failUnless(np.allclose(m1._objective(m1.get_free_state())[0], m2._objective(m2.get_free_state())[0]))

    def test_notwhite(self):
        m1 = GPflow.svgp.SVGP(self.X,
                              self.Y,
                              kern=GPflow.kernels.RBF(1) + \
                                   GPflow.kernels.White(1),
                              likelihood=GPflow.likelihoods.Exponential(),
                              Z=self.Z,
                              q_diag=True,
                              whiten=False)
        m2 = GPflow.svgp.SVGP(self.X,
                              self.Y,
                              kern=GPflow.kernels.RBF(1) + \
                                   GPflow.kernels.White(1),
                              likelihood=GPflow.likelihoods.Exponential(),
                              Z=self.Z,
                              q_diag=False,
                              whiten=False)
        m1._compile()
        m2._compile()

        qsqrt, qmean = self.rng.randn(2, 3, 2)
        qsqrt = (qsqrt**2)*0.01
        m1.q_sqrt = qsqrt
        m1.q_mu = qmean
        m2.q_sqrt = np.array([np.diag(qsqrt[:,0]), np.diag(qsqrt[:,1])]).swapaxes(0,2)
        m2.q_mu = qmean
        self.failUnless(np.allclose(m1._objective(m1.get_free_state())[0], m2._objective(m2.get_free_state())[0]))

class TestSparseMCMC(unittest.TestCase):
    """
    This test makes sure that when the inducing points are the same as the data
    points, the sparse mcmc is the same as full mcmc
    """
    def setUp(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10,1)
        Y = rng.randn(10,1)
        v_vals = rng.randn(10,1)

        l = GPflow.likelihoods.StudentT()
        self.m1 = GPflow.gpmc.GPMC(X=X, Y=Y, kern=GPflow.kernels.Exponential(1), likelihood=l)
        self.m2 = GPflow.sgpmc.SGPMC(X=X, Y=Y, kern=GPflow.kernels.Exponential(1), likelihood=l, Z=X.copy())

        self.m1.V = v_vals
        self.m2.V = v_vals.copy()
        self.m1.kern.lengthscale = .8
        self.m2.kern.lengthscale = .8
        self.m1.kern.variance = 4.2
        self.m2.kern.variance = 4.2

        self.m1._compile()
        self.m2._compile()

    def test_likelihoods(self):
        f1, _ = self.m1._objective(self.m1.get_free_state())
        f2, _ = self.m2._objective(self.m2.get_free_state())
        self.failUnless(np.allclose(f1, f2))

    def test_gradients(self):
        #the parameters might not be in the same order, so sort the gradients before checking they're the same
        _, g1 = self.m1._objective(self.m1.get_free_state())
        _, g2 = self.m2._objective(self.m2.get_free_state())
        g1 = np.sort(g1)
        g2 = np.sort(g2)
        self.failUnless(np.allclose(g1, g2))


class TestWarpedGP(unittest.TestCase):
    """
    This includes tests for warping functions, a toy model assigning
    an Identity Function to a Warped GP (should give the same results
    as a standard GP) and Snelson et. al (2004) original example
    on the "cubic sine" function.
    """
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.randn(10,1)
        self.Y = rng.randn(10,1)

    #@unittest.skip('')
    def test_wgp_identity(self):
        k = GPflow.kernels.RBF(1)
        gp = GPflow.gpr.GPR(self.X, self.Y, k)
        gp.optimize()
        gp_preds = gp.predict_y(self.X)
        
        wk = GPflow.kernels.RBF(1)
        warp = GPflow.warping_functions.IdentityFunction()
        wgp = GPflow.warped_gp.WarpedGP(self.X, self.Y, wk, warp=warp)
        wgp.optimize()
        wgp_preds = wgp.predict_y(self.X)
        
        self.failUnless(np.allclose(gp_preds, wgp_preds))

    def test_wgp_log(self):
        """
        Important catch here: a standard GP with log labels
        should have the same *median* predictions as
        a WGP with a LogFunction, not the same *mean* predictions.
        """
        k = GPflow.kernels.RBF(1)
        Y = np.abs(self.Y)
        logY = np.log(Y)
        gp = GPflow.gpr.GPR(self.X, logY, k)
        gp.optimize()
        gp_preds = gp.predict_y(self.X)
        
        wk = GPflow.kernels.RBF(1)
        warp = GPflow.warping_functions.LogFunction()
        wgp = GPflow.warped_gp.WarpedGP(self.X, Y, wk, warp=warp)
        wgp.median = True # Predicts the mean otherwise
        wgp.optimize()
        wgp_preds = wgp.predict_y(self.X)

        self.failUnless(np.allclose(np.exp(gp_preds)[0], wgp_preds[0]))



if __name__ == "__main__":
    unittest.main()

