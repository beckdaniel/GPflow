import tensorflow as tf
from .tf_hacks import eye

def gauss_kl_white(q_mu, q_sqrt, num_latent):
    """
    Compute the KL divergence from 

          q(x) = N(q_mu, q_sqrt^2)
    to 
          p(x) = N(0, I)

    We assume num_latent independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt. 

    q_mu is a matrix, each column contains a mean

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance. 

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu andthe last dim of q_sqrt). 
    """
    KL = 0.5*tf.reduce_sum(tf.square(q_mu)) #Mahalanobis term
    KL += -0.5*tf.cast(tf.shape(q_sqrt)[0]*num_latent, tf.float64) #Constant term.
    for d in range(num_latent):
        Lq = tf.user_ops.triangle(q_sqrt[:,:,d], 'lower')
        KL -= 0.5*tf.reduce_sum(tf.log(tf.square(tf.user_ops.get_diag(Lq)))) #Log determinant of q covariance.
        KL +=  0.5*tf.reduce_sum(tf.square(Lq))  #Trace term.
    return KL

def gauss_kl_white_diag(q_mu, q_sqrt, num_latent):
    """
    Compute the KL divergence from 

          q(x) = N(q_mu, q_sqrt^2)
    to 
          p(x) = N(0, I)

    We assume num_latent independent distributions, given by the columns of
    q_mu and q_sqrt. 

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each columnt represents the diagonal of a square-root
        matrix of the covariance. 

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu and q_sqrt). 
    """
 
    KL = 0.5*tf.reduce_sum(tf.square(q_mu)) #Mahalanobis term
    KL += -0.5*tf.cast(tf.shape(q_sqrt)[0]*num_latent, tf.float64) #Constant term.
    KL += -0.5*tf.reduce_sum(tf.log(tf.square(q_sqrt)))#Log determinant of q covariance.
    KL += 0.5*tf.reduce_sum(tf.square(q_sqrt)) # Trace term
    return KL


def gauss_kl_diag(q_mu, q_sqrt, K,  num_latent):
    """
    Compute the KL divergence from 

          q(x) = N(q_mu, q_sqrt^2)
    to 
          p(x) = N(0, K)

    We assume num_latent independent distributions, given by the columns of
    q_mu and q_sqrt. 

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each columnt represents the diagonal of a square-root
        matrix of the covariance of q.
 
    K is a positive definite matrix: the covariance of p.

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu and q_sqrt). 
    """
    L = tf.cholesky(K)
    alpha = tf.user_ops.triangular_solve(L, q_mu, 'lower')
    KL = 0.5*tf.reduce_sum(tf.square(alpha)) #Mahalanobis term.
    KL += num_latent * 0.5*tf.reduce_sum(tf.log(tf.square(tf.user_ops.get_diag(L) ))) #Prior log determinant term.
    KL += -0.5*tf.cast(tf.shape(q_sqrt)[0]*num_latent, tf.float64) #Constant term.
    KL += -0.5*tf.reduce_sum(tf.log(tf.square(q_sqrt))) #Log determinant of q covariance. 
    L_inv = tf.user_ops.triangular_solve(L, eye(tf.shape(L)[0]), 'lower')
    K_inv = tf.user_ops.triangular_solve(tf.transpose(L), L_inv, 'upper')
    KL += 0.5 * tf.reduce_sum(tf.expand_dims(tf.user_ops.get_diag(K_inv), 1) * tf.square(q_sqrt)) #Trace term.
    return KL

def gauss_kl(q_mu, q_sqrt, K, num_latent):
    """
    Compute the KL divergence from 

          q(x) = N(q_mu, q_sqrt^2)
    to 
          p(x) = N(0, K)

    We assume num_latent independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt. 

    q_mu is a matrix, each column contains a mean.

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance of q. 

    K is a positive definite matrix: the covariance of p.

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu andthe last dim of q_sqrt). 
    """
    L = tf.cholesky(K)
    alpha = tf.user_ops.triangular_solve(L, q_mu, 'lower')
    KL = 0.5*tf.reduce_sum(tf.square(alpha)) #Mahalanobis term.
    KL += num_latent * 0.5*tf.reduce_sum(tf.log(tf.square(tf.user_ops.get_diag(L) ))) #Prior log determinant term.
    KL += -0.5*tf.cast(tf.shape(q_sqrt)[0]*num_latent, tf.float64) #Constant term.
    for d in range(num_latent):
        Lq = tf.user_ops.triangle(q_sqrt[:,:,d], 'lower')
        KL+= -0.5*tf.reduce_sum(tf.log(tf.square(tf.user_ops.get_diag(Lq)))) #Log determinant of q covariance. 
        LiLq = tf.user_ops.triangular_solve(L, Lq, 'lower')
        KL += 0.5*tf.reduce_sum(tf.square(LiLq)) #Trace term
    return KL



