from typing import Optional

import math
import numpy as np
import tensorflow as tf
from scipy.special import erf

from .ParamValidator import ParamKind, ParamValidator

class FourierFuzzySet:
    def __init__(
        self,
        n: int,
        L: float,
        resolution: int = 200,
    ):
        ParamValidator.init(n, L)

        self.n = n
        self.a = 0
        self.b = L
        self.kernel_size = 2 * n + 1
        self.zero_idx = n

        self.complex_dtype_np = np.complex128
        self.complex_dtype_tf = tf.complex128

        # k and L need to be complex for calculations
        self.k = tf.cast(tf.range(-n, n + 1), self.complex_dtype_tf)   # (2*N_HARMONICS+1,)
        self.L = tf.cast(L, self.complex_dtype_tf)
        # pre-calculate sample points to evaluate fuzzy sets with
        self.x = tf.cast(tf.range(
            start=  self.a,
            limit=  self.b,
            delta=  L / resolution,
        ), self.complex_dtype_tf)

        # pre-calculate phase terms
        self.phase = 2j * np.pi * self.k / self.L

        idx = tf.range(tf.shape(self.k)[0])
        self.divisor = tf.where(idx == self.zero_idx, tf.ones_like(self.k), self.k)
        self.divisor = self.divisor[None, :]

        self.validate = ParamValidator(self.kernel_size)


    def get_gaussian_harmonics_batch(
        self,
        components: tf.Tensor,
        rescale: bool = False
    ) -> tf.Tensor:
        # convert to numpy for compatibility
        phase = self.phase.numpy()

        # components: (batch_size, 2)
        # self.validate(components, ParamKind.COMPONENTS)
        components_np = tf.transpose(components).numpy()
        mu = components_np[0][:, None]            # (N_KERNELS, 1)
        sigma = components_np[1][:, None]         # (N_KERNELS, 1)
        
        zk = (phase * sigma**2 - mu) / (np.sqrt(2) * sigma)

        # use scipy's erf
        erf_terms = (
            erf((self.b / (np.sqrt(2) * sigma)) + zk)
            - erf((self.a / (np.sqrt(2) * sigma)) + zk)
        )
        c_vals = (
            (sigma / self.L)
            * np.sqrt(np.pi / 2)
            * np.exp(0.5 * phase**2 * sigma**2 - phase * mu)
            * erf_terms
        )

        if rescale:
            # k=0 term = ∫₀¹ f(t) dt. For batch, this is for each kernel.
            c0 = c_vals[:, self.n] # Shape (N_KERNELS,)
            # Rescale each kernel's harmonics such that max(f(x)) == 1
            c_vals = c_vals / c0[:, None]

        return tf.convert_to_tensor(c_vals, dtype=self.complex_dtype_tf)


    @tf.function(jit_compile=True)
    def evaluate_batch(self, a: tf.Tensor) -> tf.Tensor:
        a = self.validate(a, ParamKind.COEFFS)

        # Compute phase: 2πikx/L
        # self.phase = 2j * np.pi * self.k_np / self.L
        phase = self.phase[:, None] * self.x[None, :]

        return tf.matmul(a, tf.exp(phase))

    @tf.function(jit_compile=True)
    def get_global_max_batch(self, a: tf.Tensor) -> tf.Tensor:
        a = self.validate(a, ParamKind.COEFFS)
        gmax = tf.reduce_sum(tf.abs(a), axis=1)
        return tf.minimum(gmax, tf.ones_like(gmax))