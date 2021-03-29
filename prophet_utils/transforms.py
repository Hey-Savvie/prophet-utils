"""Data transformations supporting using Prophet on bounded data."""
import abc

import numpy as np
import pandas as pd
from scipy import special


class Transform(abc.ABC):
    """Abstract interface to data transformation used to help Prophet forecast in bounded domains.

    Converts bounded real data to and from a "working" space in which they are unbounded.
    Once the data are in the working space, a Prophet model can be trained and used for forecasting.
    The forecasts need to be converted from the working space back to real space by the same instance
    of `Transform` class.

    Implementations must ensure that the result of converting finite real data to work space does not produce
    NaNs or infinities. Transformation from real to work space must be strictly order-preserving.
    """

    @abc.abstractmethod
    def to_work_series(self, data: pd.Series) -> pd.Series:
        """Converts data from real space to work space.

        Raises:
            ValueError if `data` do not respect the lower and upper bound.
        """
        ...

    @abc.abstractmethod
    def to_real_series(self, data: pd.Series) -> pd.Series:
        """Converts data from work space to real space."""
        ...

    @property
    @abc.abstractmethod
    def lower_bound(self) -> float:
        """Lower bound for real data."""
        ...

    @property
    @abc.abstractmethod
    def upper_bound(self) -> float:
        """Upper bound for real data."""
        ...


class Logarithmic(Transform):
    """Transforms non-negative data to/from unbounded representation using a shifted log-transform.

    Given eps > 0,

    Y_work = ln(eps + Y_real)

    Y_real = max(exp(Y_work) - eps, 0)

    """

    def __init__(self, eps: float):
        """Constructor.

        Args:
            eps: Positive constant added to real data before calculating the logarithm, to
                avoid producing -Infinity from finite inputs.
        """
        super().__init__()
        if not (eps > 0):
            raise ValueError(f'Epsilon must be positive, got {eps}')
        self._eps = eps

    def to_work_series(self, data: pd.Series) -> pd.Series:
        if not (np.amin(data) >= self.lower_bound):
            raise ValueError('Real data out of bounds')
        return np.log(self._eps + data)

    def to_real_series(self, data: pd.Series) -> pd.Series:
        return (np.exp(data) - self._eps).clip(self.lower_bound, None)

    @property
    def lower_bound(self) -> float:
        return 0

    @property
    def upper_bound(self) -> float:
        return np.inf


class Logit(Transform):
    """Transforms data in [0, 1] range to/from unbounded representation using a "compressed" logit transform.

    Given 0 < eps << 1/2,

    Y_work = logit( eps + Y_real * (1 - 2 * eps) )

    Y_real = min( max( (expit(Y_work) - eps) / (1 - 2 * eps), 0), 1)

    where logit(p) = ln( p / (1 - p) ) and expit(x) = 1 / (1 + exp(-x)).

    """

    def __init__(self, eps: float):
        """Constructor.

        Args:
            eps: Used compress the data range from [0, 1] to [eps, 1 - eps], so that the logit transform does not yield
                +/- Infinity on valid data.
        """
        super().__init__()
        if not (eps > 0):
            raise ValueError(f'Epsilon must be positive, got {eps}')
        if not (eps < 0.5):
            raise ValueError(f'Epsilon must be < 1/2, got {eps}')
        self._eps = eps
        self._width = 1 - 2 * eps

    def to_work_series(self, data: pd.Series) -> pd.Series:
        if not (np.amin(data) >= self.lower_bound and np.amax(data) <= self.upper_bound):
            raise ValueError('Real data out of bounds')
        return special.logit(self._eps + data * self._width)

    def to_real_series(self, data: pd.Series) -> pd.Series:
        return ((special.expit(data) - self._eps) / self._width).clip(self.lower_bound, self.upper_bound)

    @property
    def lower_bound(self) -> float:
        return 0

    @property
    def upper_bound(self) -> float:
        return 1
