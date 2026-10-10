import tensorflow as tf

from .FourierFuzzySet import FourierFuzzySet
from .ParamValidator import ParamValidator, ParamKind

class SetOps:
    """Fuzzy negation / intersection / union in Fourier coefficient space."""

    def __init__(
        self,
        n: int = 20,
        L: float = 1.0,
        dtype: tf.DType = tf.float32,
    ):
        ParamValidator.init(n=n, L=L)

        self.n = int(n)
        self.L = float(L)
        self.dtype = dtype

        self.kernel_size = 2 * self.n + 1
        self.complex_dtype = tf.complex128

        self.validator = ParamValidator(
            kernel_size=self.kernel_size
        )

        self.k_vals = tf.cast(
            tf.range(-self.n, self.n + 1),
            dtype=self.complex_dtype,
        )


    def _linear_convolve_batch(
        self,
        a: tf.Tensor,
        b: tf.Tensor,
    ) -> tf.Tensor:
        """
        Batched linear convolution using FFT.

        Parameters
        ----------
        a:
            Tensor of shape (batch_size, kernel_size).

        b:
            Tensor of shape (batch_size, kernel_size).

        Returns
        -------
        tf.Tensor
            Full convolution of shape (batch_size, 2 * kernel_size - 1).
        """
        a, b = self.validator.coeff_pair(a, b)

        fft_size = 2 * self.kernel_size - 1
        pad_width = fft_size - self.kernel_size

        a_padded = tf.pad(a, [[0, 0], [0, pad_width]])
        b_padded = tf.pad(b, [[0, 0], [0, pad_width]])

        a_fft = tf.signal.fft(a_padded)
        b_fft = tf.signal.fft(b_padded)

        return tf.signal.ifft(a_fft * b_fft)


    def _center_trim_batch(
        self,
        full_conv: tf.Tensor,
    ) -> tf.Tensor:
        """
        Trim full convolution back to 2n + 1 centered Fourier coefficients.
        """
        return full_conv[:, self.n : self.n + self.kernel_size]


    def negation_batch(
        self,
        c_vals: tf.Tensor,
    ) -> tf.Tensor:
        """
        Fuzzy negation for batched coefficient vectors.

        Parameters
        ----------
        c_vals:
            Tensor of shape (batch_size, kernel_size).

        Returns
        -------
        tf.Tensor
            Tensor of shape (batch_size, kernel_size).
        """
        c_vals = self.validator(c_vals, ParamKind.COEFFS)

        dc_impulse = tf.one_hot(
            self.n,
            self.kernel_size,
            dtype=c_vals.dtype,
        )

        return -c_vals + dc_impulse


    def intersect_batch(
        self,
        c_vals1: tf.Tensor,
        c_vals2: tf.Tensor,
        rescale: bool = False,
    ) -> tf.Tensor:
        """
        Fuzzy intersection for batched coefficient vectors.

        Parameters
        ----------
        c_vals1:
            Tensor of shape (batch_size, kernel_size).

        c_vals2:
            Tensor of shape (batch_size, kernel_size).

        rescale:
            If True, normalize each row by the sum of absolute coefficient
            magnitudes when that sum is positive.

        Returns
        -------
        tf.Tensor
            Tensor of shape (batch_size, kernel_size).
        """
        c_vals1, c_vals2 = self.validator.coeff_pair(c_vals1, c_vals2)

        full_conv = self._linear_convolve_batch(c_vals1, c_vals2)
        intersection = self._center_trim_batch(full_conv)

        if rescale:
            intersection = self._rescale_batch(intersection)

        return intersection


    def union_batch(
        self,
        c_vals1: tf.Tensor,
        c_vals2: tf.Tensor,
    ) -> tf.Tensor:
        """
        Fuzzy union for batched coefficient vectors.

        Parameters
        ----------
        c_vals1:
            Tensor of shape (batch_size, kernel_size).

        c_vals2:
            Tensor of shape (batch_size, kernel_size).

        Returns
        -------
        tf.Tensor
            Tensor of shape (batch_size, kernel_size).
        """
        c_vals1, c_vals2 = self.validator.coeff_pair(c_vals1, c_vals2)

        return c_vals1 + c_vals2 - self.intersect_batch(c_vals1, c_vals2)

    @staticmethod
    def _rescale_batch(
        x: tf.Tensor,
    ) -> tf.Tensor:
        """
        Normalize each row by sum(abs(row)) when that sum is positive.
        """
        s = tf.reduce_sum(
            tf.abs(x),
            axis=-1,
            keepdims=True,
        )

        # Keep s as real for comparison, cast to complex for division
        s_complex = tf.cast(s, x.dtype)

        # Compare with real zero, then use complex s for division
        return tf.where(
            s > 0,
            x / s_complex,
            x,
        )