from enum import Enum, auto

import numpy as np
import tensorflow as tf


class ParamKind(Enum):
    """Supported parameter validation modes."""

    COMPONENTS = auto()
    COEFFS = auto()
    COEFF_VECTOR = auto()
    COEFF_PAIR = auto()
    COEFF_VECTOR_PAIR = auto()


class ParamValidator:
    """Validate component parameters and Fourier coefficients."""

    def __init__(self, kernel_size: int):
        """Store expected kernel size and dtypes."""
        self.kernel_size = int(kernel_size)
        self.complex_dtype = tf.complex128


    @staticmethod
    def init(n: int, L: float) -> None:
        """Validate initialization arguments."""
        if n < 0:
            raise ValueError(f"n must be nonnegative, got {n}")
        if not np.isfinite(L) or L == 0:
            raise ValueError(f"L must be finite and nonzero, got {L}")


    def __call__(self, x, kind: ParamKind) -> tf.Tensor:
        """Validate one argument according to its parameter kind."""
        handlers = {
            ParamKind.COMPONENTS: self.components,
            ParamKind.COEFFS: self.coeffs,
            ParamKind.COEFF_VECTOR: self.coeff_vector,
        }

        try:
            return handlers[kind](x)
        except KeyError:
            raise ValueError(f"Unsupported ParamKind for single argument: {kind}")


    def components(self, x) -> tf.Tensor:
        """Validate components with shape `(batch_size, 2)` and positive sigma."""
        x = self._validate(
            x,
            name="components",
            shape=(None, 2),
            cast=True,
        )
        tf.debugging.assert_positive(x[:, 1], message="all sigma values must be positive")
        return x


    def coeffs(self, x) -> tf.Tensor:
        """Validate batched coefficients with shape `(batch_size, kernel_size)`."""
        return self._validate(
            x,
            name="coeffs",
            shape=(None, self.kernel_size),
        )


    def coeff_vector(self, x) -> tf.Tensor:
        """Validate one coefficient vector with shape `(kernel_size,)`."""
        return self._validate(
            x,
            name="coeff_vector",
            shape=(self.kernel_size,),
        )


    def coeff_pair(self, x1, x2) -> tuple[tf.Tensor, tf.Tensor]:
        """Validate two batched coefficient tensors with matching shapes."""
        return self._pair(self.coeffs, x1, x2, "coeffs1", "coeffs2")


    def coeff_vector_pair(self, x1, x2) -> tuple[tf.Tensor, tf.Tensor]:
        """Validate two coefficient vectors with matching shapes."""
        return self._pair(
            self.coeff_vector,
            x1,
            x2,
            "coeff_vector1",
            "coeff_vector2",
        )


    def coeff_vector_to_batch(self, x) -> tf.Tensor:
        """Validate one coefficient vector and add a batch dimension."""
        return tf.expand_dims(self.coeff_vector(x), axis=0)


    def coeff_vector_pair_to_batch(self, x1, x2) -> tuple[tf.Tensor, tf.Tensor]:
        """Validate two coefficient vectors and add batch dimensions."""
        x1, x2 = self.coeff_vector_pair(x1, x2)
        return tf.expand_dims(x1, axis=0), tf.expand_dims(x2, axis=0)


    @staticmethod
    def components_np(x) -> np.ndarray:
        """Validate NumPy components with shape `(batch_size, 2)`."""
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"components must have shape (batch_size, 2), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("components must contain only finite values")
        if np.any(x[:, 1] <= 0):
            raise ValueError("all sigma values must be positive")

        return x


    @staticmethod
    def _pair(fn, x1, x2, name1: str, name2: str) -> tuple[tf.Tensor, tf.Tensor]:
        """Validate two tensors with the same validator and shape."""
        x1, x2 = fn(x1), fn(x2)
        tf.debugging.assert_equal(
            tf.shape(x1),
            tf.shape(x2),
            message=f"{name1} and {name2} must have the same shape",
        )
        return x1, x2


    @staticmethod
    def _validate(
        x,
        *,
        name: str,
        shape: tuple[int | None, ...],
        cast: bool = False,
    ) -> tf.Tensor:
        """Validate tensor dtype, rank, selected dimensions, and finiteness."""
        x = tf.convert_to_tensor(x) if cast else tf.convert_to_tensor(x)

        tf.debugging.assert_rank(
            x,
            len(shape),
            message=f"{name} must be rank {len(shape)}",
        )

        for axis, expected in enumerate(shape):
            if expected is not None:
                tf.debugging.assert_equal(
                    tf.shape(x)[axis],
                    expected,
                    message=f"{name} dimension {axis} must be {expected}",
                )

        ParamValidator._assert_finite(x, name)
        return x


    @staticmethod
    def _assert_finite(x: tf.Tensor, name: str) -> None:
        """Assert all real and imaginary values are finite."""
        tf.debugging.assert_all_finite(
            tf.math.real(x),
            message=f"real({name}) contains non-finite values",
        )

        if x.dtype.is_complex:
            tf.debugging.assert_all_finite(
                tf.math.imag(x),
                message=f"imag({name}) contains non-finite values",
            )