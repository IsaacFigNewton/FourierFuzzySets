
"""
Fourier Fuzzy GMM - build and evaluate a simple Gaussian Mixture Model
using the Fourier-series fuzzy set framework from:

  "Fourier Series Approximations of Likelihood-Based Fuzzy Sets"
   (Rudnick, 2026)
"""
import tensorflow as tf
from .FourierFuzzySet import FourierFuzzySet

class Mixture:
    def __init__(self, model: FourierFuzzySet) -> None:
        self.model = model

    def fourier_gmm(self, ops, components:tf.Tensor, weights:tf.Tensor) -> tf.Tensor:
        """
        Construct a mixture by weighted fuzzy-union of Fourier kernels.

        For a simple additive mixture we scale each kernel's coefficients
        by its weight and sum - equivalent to the analytic mixture.

        params:
        components: an tf.Tensor of shape (n, 2) where n is number of kernels
        weights: an tf.Tensor of shape (n,)
        """
        # c_add:          [N_KERNELS, 2*N_HARMONICS+1]
        c_add = self.model.get_gaussian_harmonics_batch(components, rescale=True)
        # For element-wise multiplication, weights needs to be (N_KERNELS, 1)
        # Cast weights to complex to match c_add dtype
        weights_complex = tf.cast(weights[:, None], c_add.dtype)
        # c_add_weighted: [N_KERNELS, 1] x [N_KERNELS, K_VALS_SIZE]
        c_add_weighted = tf.broadcast_to(weights_complex, c_add.shape) * c_add
        return tf.reduce_sum(c_add_weighted, axis=0)